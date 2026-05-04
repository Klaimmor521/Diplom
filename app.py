import streamlit as st
import seaborn as sns
import numpy as np
import pandas as pd
import hashlib

from modules.data_processor import load_data, classify_smart
from modules.charts import (
    analyze_profitability,
    draw_abc_pie_chart,
    draw_revenue_bar, 
    draw_top_items_pie, 
    draw_yoy_chart, 
    draw_forecast_chart, 
    perform_abc_analysis, 
    draw_seasonality_chart,
)
from modules.ml_model import run_prediction

sns.set_theme(
    style="whitegrid",
    rc={
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "Verdana"],
        "axes.titlesize": 16,
        "axes.titleweight": "bold",
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "legend.title_fontsize": 11,
        "figure.dpi": 150,
    }
)

# --- Настройки ---
st.set_page_config(page_title="Прогнозирование спроса", layout="wide", initial_sidebar_state="expanded")
st.title("Система прогнозирования и аналитики")

# --- Загрузка ---
st.sidebar.header("🗂️ Данные")
uploaded_files = st.sidebar.file_uploader(
    "Загрузите один или несколько отчетов",
    type=["csv", "xlsx"],
    accept_multiple_files=True
)

# Создаем пустой DataFrame, который будем наполнять
df = None
data_frames = []
seen_hashes = set()

if uploaded_files:
    # Если пользователь загрузил файлы, читаем и собираем их
    for file in uploaded_files:
        # Считаем хеш содержимого
        file_bytes = file.read()
        file_hash = hashlib.md5(file_bytes).hexdigest()
        file.seek(0)  # возвращаем курсор в начало для чтения в load_data
        
        if file_hash in seen_hashes:
            st.warning(f"Файл '{file.name}' уже был загружен (по содержимому). Пропускаем дубликат.")
            continue
        seen_hashes.add(file_hash)
        
        df = load_data(file)
        if df is not None:
            data_frames.append(df)
else:
    test_files = ["data/livesklad_report_2023.csv", "data/livesklad_report_2024.csv", "data/livesklad_report_2025.csv"]
    for file_path in test_files:
        try:
            df = load_data(file_path)
            if df is not None:
                data_frames.append(df)
        except FileNotFoundError:
            pass # Игнорируем, если тестового файла нет

# Если удалось собрать хотя бы один DataFrame, объединяем их
if data_frames:
    df = pd.concat(data_frames, ignore_index=True)

if df is None:
    st.warning("Пожалуйста, загрузите файл(ы) с данными для начала анализа.")
    st.stop()

# --- Фильтры---
st.sidebar.header("⚙️ Настройки фильтрации")

all_years = sorted(df['Дата'].dt.year.unique(), reverse=True)
if not all_years:
    st.error("В файле не найдены данные с корректными датами.")
    st.stop()

selected_year = st.sidebar.selectbox(
    "Выберите год для анализа:",
    options=all_years,
    index=0,
    help="Этот год будет использоваться для всех отчетов, кроме 'Сравнения по годам'."
)

target_type = st.sidebar.radio(
    "Основной показатель для анализа:",
    ["Выручка (₽)", "Количество (шт)"],
    help="""
    Определяет единицы измерения для всех отчетов:

    - **Выручка (₽):** Ключевой показатель для оценки финансовой эффективности. **Рекомендуется для анализа услуг.**

    - **Количество (шт):** Важный показатель для управления запасами товаров. Для услуг эта метрика отражает **количество выполненных операций (транзакций)** и полезна для оценки потока клиентов или популярности конкретной услуги.
    """
)

target_col = "Сумма" if target_type == "Выручка (₽)" else "Количество"

st.sidebar.divider()
use_smart_sort = st.sidebar.checkbox(
    "Применить смарт-категоризацию", 
    value=True, 
    help="Активирует алгоритм автоматического распределения номенклатуры по бизнес-категориям (Услуги, Запчасти, Аксессуары)."
)

df['Категория'] = df.apply(lambda row: classify_smart(row, use_smart_sort), axis=1)
unique_categories = sorted(df['Категория'].unique())
selected_type = st.sidebar.multiselect(
    "Фильтр по категориям:", 
    options=unique_categories, 
    default=unique_categories,
    help="Выберите одну или несколько категорий для построения отчетов."
)

if not selected_type:
    st.warning("Пожалуйста, выберите хотя бы одну категорию.")
    st.stop()
    
df_filtered = df[df['Категория'].isin(selected_type)]

unique_items = sorted(df_filtered['Название'].unique())
selected_items = st.sidebar.multiselect(
    "Фильтр по номенклатуре:", 
    options=unique_items,
    default=[],
    help="Оставьте поле пустым для анализа всей категории или выберите конкретные позиции для детального отчета."
)

if len(selected_items) > 0:
    df_filtered = df_filtered[df_filtered['Название'].isin(selected_items)]

df_filtered_single_year = df[df['Дата'].dt.year == selected_year]
df_filtered_single_year = df_filtered_single_year[df_filtered_single_year['Категория'].isin(selected_type)]
if selected_items:
    df_filtered_single_year = df_filtered_single_year[df_filtered_single_year['Название'].isin(selected_items)]

if df_filtered_single_year.empty:
    st.warning("По выбранным фильтрам нет данных. Попробуйте изменить выбор.")
    st.stop()

# --- Экономика ---
st.sidebar.divider()
st.sidebar.header("💼 Экономика бизнеса")
master_percent = st.sidebar.slider(
    "Процент ЗП мастера с услуг (%)", 
    min_value=0, max_value=100, value=40,
    help="Укажите процент сдельной оплаты мастеров для расчета чистой прибыли."
)

# --- Интерфейс ---
st.subheader("Финансовые показатели")

total_revenue = df_filtered_single_year['Сумма'].sum()
total_gross_profit = df_filtered_single_year['Валовая прибыль (руб)'].sum()

# Всегда считаем чистую прибыль с учётом процента на услуги
service_revenue = df_filtered_single_year[df_filtered_single_year['Категория'] == 'Услуги']['Сумма'].sum()
net_profit = total_gross_profit - (service_revenue * (master_percent / 100))

col1, col2, col3 = st.columns(3)
col1.metric("Оборот (Выручка)", f"{total_revenue:,.0f} ₽")
col2.metric("Валовая прибыль", f"{total_gross_profit:,.0f} ₽")
col3.metric("Чистая прибыль (после ЗП)", f"{net_profit:,.0f} ₽", 
            help=f"Валовая прибыль за вычетом {master_percent}% от выручки услуг на зарплату мастеров")

st.divider()

# --- Вкладки ---
tab1, tab2, tab3 = st.tabs(["Общий обзор", "Детальный анализ", "Прогнозирование"])

# --- Вкладка 1: общий обзор ---
with tab1:
    st.subheader(f"Анализ за {selected_year} год")
    col_left, col_right = st.columns(2)
    with col_left:
        st.subheader("Динамика по месяцам", help="Показывает исторические продажи (в ₽ или шт). Помогает оценить сезонность.")
        st.pyplot(draw_revenue_bar(df_filtered_single_year, target_col))
    with col_right:
        st.subheader("Топ-10 Популярных позиций", help="Рейтинг самых продаваемых позиций в деньгах или штуках.")
        st.pyplot(draw_top_items_pie(df_filtered_single_year, target_col))

    st.divider()
    st.subheader("Сравнение продаж: Год к Году", help="Сравнивает продажи одних и тех же месяцев в разные годы.")

    if len(all_years) < 2:
        st.info("Для сравнения необходимо иметь данные как минимум за два года.")
    else:
        yoy_col1, yoy_col2 = st.columns(2)
        with yoy_col1:
            year1 = st.selectbox("Выберите первый год:", options=all_years, index=1)
        with yoy_col2:
            year2 = st.selectbox("Выберите второй год:", options=all_years, index=0)

        if year1 == year2:
            st.warning("Пожалуйста, выберите два разных года для сравнения.")
        else:
            # Фильтруем полный датасет (df) по двум выбранным годам
            df_yoy = df[df['Дата'].dt.year.isin([year1, year2])]
            st.pyplot(draw_yoy_chart(df_yoy, target_col))

# --- Вкладка 2: детальный анализ ---
with tab2:
    st.subheader(f"Детальный анализ за {selected_year} год")
    st.subheader("Детальный отчет")
    with st.expander("Открыть таблицу с данными"):
        st.dataframe(df_filtered_single_year, width='stretch')
    
    st.divider()
    st.subheader("ABC-анализ номенклатуры", help="Делит все позиции на 3 группы по их вкладу в общую выручку. Анализируется только по Выручке.")

    abc_df = perform_abc_analysis(df_filtered_single_year)

    if not abc_df.empty:
        total_items = len(abc_df)
        total_revenue = df_filtered_single_year['Сумма'].sum()
        
        # Расчеты для каждой группы
        group_a = abc_df[abc_df['Категория'] == 'A']
        group_b = abc_df[abc_df['Категория'] == 'B']
        group_c = abc_df[abc_df['Категория'] == 'C']

        a_items, a_revenue = len(group_a), group_a['Выручка'].sum()
        b_items, b_revenue = len(group_b), group_b['Выручка'].sum()
        c_items, c_revenue = len(group_c), group_c['Выручка'].sum()

        # Отображение метрик в три колонки
        m_col1, m_col2, m_col3 = st.columns(3)
        with m_col1:
            st.metric(
                label="Группа А (Ключевые)",
                value=f"{a_items} поз. ({(a_items/total_items)*100:.0f}%)",
                delta=None,
                delta_color="off",
                help="Обычно это 10-20% ассортимента, которые приносят ~80% всей выручки. Самые важные позиции, требующие максимального внимания."
            )
            st.markdown(f"**Вклад в выручку:** <span style='color: #10B981;'>{a_revenue:,.0f} ₽ ({(a_revenue/total_revenue)*100:.2f}%)</span>", unsafe_allow_html=True)

        with m_col2:
            st.metric(
                label="Группа B (Стабильные)",
                value=f"{b_items} поз. ({(b_items/total_items)*100:.0f}%)",
                delta=None,
                delta_color="off",
                help="Промежуточная группа: добавляет ещё ~15% выручки после лидеров. Эти позиции важны, но не критичны."
            )
            st.markdown(f"**Вклад в выручку:** {b_revenue:,.0f} ₽ ({(b_revenue/total_revenue)*100:.2f}%)")

        with m_col3:
            st.metric(
                label="Группа C (Незначительные)",
                value=f"{c_items} поз. ({(c_items/total_items)*100:.0f}%)",
                delta=None,
                delta_color="off",
                help="Всё, что осталось после группы B. Обычно самая многочисленная часть ассортимента, но даёт лишь ~5% выручки. Кандидаты на оптимизацию."
            )
            st.markdown(f"**Вклад в выручку:** <span style='color: #f44336;'>{c_revenue:,.0f} ₽ ({(c_revenue/total_revenue)*100:.2f}%)</span>", unsafe_allow_html=True)
        
        st.markdown("---")

        # График и вкладки с таблицами
        col_pie, col_tabs = st.columns([1, 2])
        with col_pie:
            st.pyplot(draw_abc_pie_chart(abc_df))
        
        with col_tabs:
            tab_a, tab_b, tab_c = st.tabs(["Группа A", "Группа B", "Группа C"])
            with tab_a:
                st.dataframe(group_a, hide_index=True, width='stretch',
                    column_config={"Категория": None, "Выручка": st.column_config.NumberColumn("Выручка, ₽", format="%.0f ₽"),
                                "Доля в %": st.column_config.ProgressColumn("Доля в выручке", format="%.2f%%", min_value=0, max_value=100)})
            with tab_b:
                st.dataframe(group_b, hide_index=True, width='stretch',
                    column_config={"Категория": None, "Выручка": st.column_config.NumberColumn("Выручка, ₽", format="%.0f ₽"),
                                "Доля в %": st.column_config.ProgressColumn("Доля в выручке", format="%.2f%%", min_value=0, max_value=100)})
            with tab_c:
                st.dataframe(group_c, hide_index=True, width='stretch',
                    column_config={"Категория": None, "Выручка": st.column_config.NumberColumn("Выручка, ₽", format="%.0f ₽"),
                                "Доля в %": st.column_config.ProgressColumn("Доля в выручке", format="%.2f%%", min_value=0, max_value=100)})
    else:
        st.info("Недостаточно данных для проведения ABC-анализа.")
        
    st.divider()
    st.subheader("Анализ сезонности", help="Показывает, в какие месяцы продажи обычно выше или ниже среднего.")
    st.pyplot(draw_seasonality_chart(df_filtered_single_year, target_col))

    st.divider()
    st.subheader("Рейтинг рентабельности", help="Показывает, какие товары и услуги наиболее эффективно превращают выручку в прибыль. Высокий % означает высокую маржинальность.")

    profit_df = analyze_profitability(df_filtered_single_year)

    if not profit_df.empty:
        col_best, col_worst = st.columns(2)

        with col_best:
            st.success("Топ-5 наиболее рентабельных")
            # Показываем и рентабельность, и число продаж
            st.dataframe(
                profit_df.head(5),
                column_config={
                    "Название": st.column_config.TextColumn("Позиция", width="large"),
                    "Средняя_рентабельность": st.column_config.NumberColumn("Рентабельность", format="%d%%"),
                    "Количество_продаж": st.column_config.NumberColumn("Продажи", format="%d"),
                },
                hide_index=True, width='stretch'
            )

        with col_worst:
            st.error("Топ-5 наименее рентабельных")
            least = profit_df.nsmallest(5, 'Средняя_рентабельность')
            st.dataframe(
                least,
                column_config={
                    "Название": st.column_config.TextColumn("Позиция", width="large"),
                    "Средняя_рентабельность": st.column_config.NumberColumn("Рентабельность", format="%d%%"),
                    "Количество_продаж": st.column_config.NumberColumn("Продажи", format="%d"),
                },
                hide_index=True, width='stretch'
            )
    else:
        st.info("В отчете отсутствуют данные для анализа рентабельности за выбранный период.")

# --- Вкладка 3: прогнозирование ---
with tab3:
    st.subheader(f"Прогноз спроса")
    use_all_data = st.checkbox(
        "Использовать все загруженные данные для обучения модели",
        value=False, # По умолчанию выключено
        help=f"""
        **По умолчанию (рекомендуется):** Прогноз строится на данных только за один выбранный год (`{selected_year}`). Это обеспечивает максимальную релевантность.

        **При активации:** Модель будет обучаться на всем доступном периоде. Это может помочь уловить долгосрочный тренд, но старые данные могут исказить прогноз, если бизнес-модель менялась.
        """
    )

    if use_all_data:
        data_for_prediction = df_filtered
        prediction_period_str = f"на основе всех лет ({', '.join(map(str, all_years))})"
    else:
        data_for_prediction = df_filtered_single_year
        prediction_period_str = f"на основе данных за {selected_year} год"

    st.markdown(f"**Построение прогноза {prediction_period_str}**")
    
    # Готовим данные для модели
    df_monthly = data_for_prediction.groupby('Месяц', as_index=False)[target_col].sum()

    if len(df_monthly) < 3:
        st.warning(f"⚠️ Недостаточно данных для прогноза в {selected_year} году. Нужно минимум 3 месяца с продажами.")
    else:
        future_months = st.slider("Горизонт прогнозирования (месяцы):", 1, 12, 6)
        
        _, y_values, future_X, future_pred, r2, mae = run_prediction(df_monthly, target_col, future_months)

        st.pyplot(draw_forecast_chart(df_monthly, df_monthly[target_col], future_X, future_pred, target_type))

        st.markdown("##### Метрики качества модели")
        col_m1, col_m2 = st.columns(2)
        col_m1.metric("Точность тренда (R²)", f"{r2:.1f}%", help="Коэффициент детерминации (R²). Показывает, какой процент дисперсии исходных данных объясняется моделью. Значение, близкое к 100%, указывает на высокую точность тренда.")
        col_m2.metric("Средняя ошибка (MAE)", f"{mae:,.0f}", help=f"Средняя абсолютная ошибка (MAE). Показывает среднее абсолютное отклонение прогнозируемых значений от фактических в тех же единицах измерения (₽ или шт).")
        
        st.divider()
        st.subheader("Аналитическая сводка и бизнес-рекомендации")
        
        avg_fact = np.mean(y_values[-3:]) if len(y_values) >= 3 else np.mean(y_values)
        avg_forecast = np.mean(future_pred)
        if abs(avg_fact) < 1e-9:
            st.warning("Фактические значения близки к нулю. Процентное изменение не рассчитывается.")
            delta = 0
        else:
            delta = ((avg_forecast - avg_fact) / avg_fact) * 100

        active_cats = set(selected_type) if selected_type else set(unique_categories)

        has_services = "Услуги" in active_cats or "Услуга" in active_cats
        has_parts = "Запчасти" in active_cats
        has_accessories = "Аксессуары" in active_cats
        has_goods = "Товар" in active_cats or "Прочее" in active_cats
        
        if delta > 10:
            with st.container():
                st.success(f"**Тренд: уверенный рост на ~{delta:.0f}%**")
                if target_col == "Количество":
                    st.markdown("""
                    **План действий (Ресурсы):**
                    - **Запасы:** :green[Увеличьте] объем закупок и сформируйте страховой запас (+20%) по ключевым позициям.
                    - **Ассортимент:** :green[Расширьте] линейку аксессуаров, особенно высокомаржинальных.
                    - **Персонал:** :green[Проверьте] загрузку мастеров и подготовьтесь к росту клиентского потока.
                    """)
                else:  # Выручка
                    st.markdown("""
                    **План действий (Финансы и Маркетинг):**
                    - **Инвестиции:** :green[Направьте] растущий денежный поток на закупку оборудования, рекламу или обучение.
                    - **Средний чек:** :green[Предлагайте] более дорогие аналоги, пакетные предложения и сопутствующие услуги.
                    """)

        elif delta < -10:
            with st.container():
                st.error(f"**Тренд: прогнозируется спад на ~{abs(delta):.0f}%**")
                if target_col == "Количество":
                    st.markdown("""
                    **План действий (Ресурсы):**
                    - **Закупки:** :red[Приостановите] новые заказы по падающим категориям и сфокусируйтесь на распродаже излишков.
                    - **Ассортимент:** :red[Сократите] линейку до самых оборачиваемых позиций, запустите акции на неликвиды.
                    - **Персонал:** :red[Запустите] маркетинговые акции (скидки, комплексные предложения) для привлечения клиентов.
                    """)
                else:  # Выручка
                    st.markdown("""
                    **План действий (Финансы и Маркетинг):**
                    - **Расходы:** :red[Проведите] аудит постоянных затрат, сократите неэффективные маркетинговые каналы.
                    - **Спрос:** :red[Пересмотрите] ценовую политику, возможно, временно снизьте наценку для удержания клиентов.
                    """)
        else:
            with st.container():
                st.info(f"**Тренд: стабильный (изменение {delta:+.1f}%)**")
                st.markdown("""
                **План действий:**
                - **Операции:** :blue[Поддерживайте] текущий уровень складских запасов и график работы персонала.
                - **Развитие:** :blue[Используйте] стабильный период для улучшения внутренних процессов, обучения сотрудников и внедрения новых стандартов качества.
                """)

        if r2 < 30:
            st.caption("⚠️ **Примечание:** Исторические данные крайне волатильны (R² < 30%). Это означает, что простой линейный тренд слабо описывает реальные продажи. Рекомендуется принимать решения с повышенной осторожностью и дробить закупки на более мелкие партии.")