import streamlit as st
import seaborn as sns
import numpy as np

from modules.data_processor import load_data, classify_smart
from modules.charts import (
    analyze_profitability,
    draw_abc_pie_chart,
    draw_revenue_bar, 
    draw_top_items_pie, 
    draw_yoy_chart, 
    draw_forecast_chart, 
    perform_abc_analysis, 
    draw_seasonality_chart
)
from modules.ml_model import run_prediction

# --- Настройки ---
st.set_page_config(page_title="Прогнозирование спроса", layout="wide", initial_sidebar_state="expanded")
st.title("Система прогнозирования и аналитики")
sns.set_theme(style="darkgrid")

# --- Загрузка ---
st.sidebar.header("📂 Данные")
uploaded_file = st.sidebar.file_uploader("Загрузите отчет LiveSklad", type=["csv", "xlsx"])

with st.spinner("Загрузка данных..."):
    if uploaded_file is not None:
        df = load_data(uploaded_file)
    else:
        try:
            df = load_data("data/Отчет по товарам и работам.xlsx")
        except FileNotFoundError:
            st.sidebar.error("Тестовый файл не найден. Пожалуйста, загрузите свой отчет.")
            st.stop()


if df is None:
    st.error("❌ Файл данных не может быть загружен. Проверьте формат файла.")
    st.stop()

# --- Фильтры---
st.sidebar.header("⚙️ Настройки фильтрации")
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

if df_filtered.empty:
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
show_net_profit = st.toggle("Расчет чистой прибыли с учетом ФОТ")

total_revenue = df_filtered['Сумма'].sum()
total_gross_profit = df_filtered['Валовая прибыль (руб)'].sum()

if show_net_profit:
    service_revenue = df_filtered[df_filtered['Категория'] == 'Услуги']['Сумма'].sum()
    final_profit = total_gross_profit - (service_revenue * (master_percent / 100))
    profit_label = "Чистая прибыль (после выплаты ЗП)"
else:
    final_profit = total_gross_profit
    profit_label = "Валовая прибыль (до выплаты ЗП)"

col1, col2 = st.columns(2)
col1.metric("Оборот (Выручка)", f"{total_revenue:,.0f} ₽")
col2.metric(profit_label, f"{final_profit:,.0f} ₽")

st.divider()

# --- Вкладки ---
tab1, tab2, tab3 = st.tabs(["Общий обзор", "Детальный анализ", "Прогнозирование"])

# --- Вкладка 1: общий обзор ---
with tab1:
    col_left, col_right = st.columns(2)
    with col_left:
        st.subheader("Динамика по месяцам", help="Показывает исторические продажи (в ₽ или шт). Помогает оценить сезонность.")
        st.pyplot(draw_revenue_bar(df_filtered, target_col)) 
    with col_right:
        st.subheader("Топ-10 Популярных позиций", help="Рейтинг самых продаваемых позиций в деньгах или штуках.")
        st.pyplot(draw_top_items_pie(df_filtered, target_col))

    st.subheader("Сравнение продаж: Год к Году", help="Сравнивает продажи одних и тех же месяцев в разные годы.")
    st.pyplot(draw_yoy_chart(df_filtered, target_col))

# --- Вкладка 2: детальный анализ ---
with tab2:
    st.subheader("Детальный отчет")
    with st.expander("Открыть таблицу с данными"):
        st.dataframe(df_filtered, width='stretch')
        st.download_button("📥 Скачать таблицу в CSV", df_filtered.to_csv(index=False, encoding='utf-8-sig'), "filtered_report.csv", "text/csv")
    
    st.divider()
    st.subheader("ABC-анализ номенклатуры", help="Делит все позиции на 3 группы по их вкладу в общую выручку. Анализируется только по Выручке.")

    abc_df = perform_abc_analysis(df_filtered)

    if not abc_df.empty:
        total_items = len(abc_df)
        total_revenue = df_filtered['Сумма'].sum()
        
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
                value=f"{a_items} поз.",
                delta=f"{a_items/total_items:.1%}",
            )
            st.markdown(f"**Вклад в выручку:** <span style='color: #10B981;'>{a_revenue:,.0f} ₽</span>", unsafe_allow_html=True)

        with m_col2:
            st.metric(
                label="Группа B (Стабильные)",
                value=f"{b_items} поз.",
                delta=f"{b_items/total_items:.1%}",
            )
            st.markdown(f"**Вклад в выручку:** {b_revenue:,.0f} ₽")

        with m_col3:
            st.metric(
                label="Группа C (Незначительные)",
                value=f"{c_items} поз.",
                delta=f"{c_items/total_items:.1%}",
                delta_color="inverse"
            )
            st.markdown(f"**Вклад в выручку:** <span style='color: #f44336;'>{c_revenue:,.0f} ₽</span>", unsafe_allow_html=True)
        
        st.markdown("---")

        # График и вкладки с таблицами
        col_pie, col_tabs = st.columns([1, 2])
        with col_pie:
            st.pyplot(draw_abc_pie_chart(abc_df))
        
        with col_tabs:
            tab_a, tab_b, tab_c = st.tabs(["Группа A", "Группа B", "Группа C"])
            with tab_a:
                st.dataframe(group_a, hide_index=True, use_container_width=True,
                    column_config={"Категория": None, "Выручка": st.column_config.NumberColumn("Выручка, ₽", format="%.0f ₽"),
                                "Доля в %": st.column_config.ProgressColumn("Доля в выручке", format="%.2f%%", min_value=0, max_value=100)})
            with tab_b:
                st.dataframe(group_b, hide_index=True, use_container_width=True,
                    column_config={"Категория": None, "Выручка": st.column_config.NumberColumn("Выручка, ₽", format="%.0f ₽"),
                                "Доля в %": st.column_config.ProgressColumn("Доля в выручке", format="%.2f%%", min_value=0, max_value=100)})
            with tab_c:
                st.dataframe(group_c, hide_index=True, use_container_width=True,
                    column_config={"Категория": None, "Выручка": st.column_config.NumberColumn("Выручка, ₽", format="%.0f ₽"),
                                "Доля в %": st.column_config.ProgressColumn("Доля в выручке", format="%.2f%%", min_value=0, max_value=100)})
    else:
        st.info("Недостаточно данных для проведения ABC-анализа.")
        
    st.divider()
    st.subheader("Анализ сезонности", help="Показывает, в какие месяцы продажи обычно выше или ниже среднего.")
    st.pyplot(draw_seasonality_chart(df_filtered, target_col))

    st.divider()
    st.subheader("Рейтинг рентабельности", help="Показывает, какие товары и услуги наиболее эффективно превращают выручку в прибыль. Высокий % означает высокую маржинальность.")

    profit_df = analyze_profitability(df_filtered)

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
                hide_index=True, use_container_width=True
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
                hide_index=True, use_container_width=True
            )
    else:
        st.info("Колонка 'Валовая прибыль (%)' отсутствует или содержит только пустые значения.")

# --- Вкладка 3: прогнозирование ---
with tab3:
    st.subheader("Прогноз спроса")
    df_monthly = df_filtered.groupby('Месяц', as_index=False)[target_col].sum()

    if len(df_monthly) < 3:
        st.warning("⚠️ Недостаточно данных для прогноза. Нужно минимум 3 месяца.")
    else:
        future_months = st.slider("Горизонт прогнозирования (месяцы):", 1, 12, 6)
        
        X, y_values, future_X, future_pred, r2, mae = run_prediction(df_monthly, target_col, future_months)

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

        types_to_check = selected_type if len(selected_type) > 0 else unique_categories
        is_services = any(cat in types_to_check for cat in ["Услуги", "Услуга"])
        is_goods = any(cat in types_to_check for cat in ["Запчасти", "Аксессуары", "Товар", "Прочее"])
        
        if delta > 10:
            st.success(f"📈 **Тренд: уверенный рост на ~{delta:.0f}%**")
            st.write("**📝 Рекомендации:**")
            if target_col == "Количество":
                if is_goods:
                    st.write("📦 **Закупки и склад:** увеличьте объём закупок пропорционально прогнозу. Сформируйте страховой запас (+15-20% к среднемесячным продажам).")
                if is_services:
                    st.write("👨‍🔧 **Персонал:** ожидается рост клиентского потока – проверьте загрузку мастеров, при необходимости расширьте штат.")
            else:  # выручка
                st.write("💰 **Бюджет:** ожидается рост денежного потока. Рекомендуется направить дополнительные средства на развитие ассортимента, маркетинг или досрочное погашение обязательств.")
        elif delta < -10:
            st.error(f"📉 **Тренд: прогнозируется спад на ~{abs(delta):.0f}%**")
            st.write("**📝 Рекомендации:**")
            if target_col == "Количество":
                if is_goods:
                    st.write("📦 **Закупки и склад:** приостановите новые заказы, распродавайте излишки. Пересмотрите минимальные остатки.")
                if is_services:
                    st.write("👨‍🔧 **Персонал:** возможна низкая загрузка – запустите акции, скидки, пересмотрите график работы.")
            else:
                st.write("💰 **Бюджет:** ожидается снижение выручки. Проведите аудит расходов, откажитесь от неэффективных затрат, усильте контроль дебиторской задолженности.")
        else:
            st.info(f"⚖️ **Тренд: стабильный (изменение {delta:+.1f}%)**")
            st.write("**📝 Рекомендации:**")
            if target_col == "Количество":
                st.write("Поддерживайте текущий уровень запасов, придерживайтесь плановых закупок.")
            else:
                st.write("Финансовые показатели стабильны. Сохраняйте текущий бюджет и инвестиционные планы.")
            
        if r2 < 30:
            st.caption("⚠️ *Примечание: Исторические данные нестабильны. Модели сложно выявить четкий тренд. Рекомендуется принимать решения с осторожностью.*")