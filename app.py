import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score

# Настройки страницы
st.set_page_config(page_title="Аналитика Сервиса", layout="wide")
st.title("📊 Система прогнозирования и аналитики")

sns.set_theme(style="whitegrid")

# Загрузка данных
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("data/livesklad_export.csv")
        
        df['Дата'] = pd.to_datetime(df['Дата'], format="%d.%m.%Y - %H:%M")
        
        df['Месяц'] = df['Дата'].dt.to_period('M').astype(str)
        return df
    except FileNotFoundError:
        return None

df = load_data()

if df is None:
    st.error("Файл данных не найден! Надо запустить data_generator.py")
    st.stop()

# --- 2. Боковая панель (Фильтры и Загрузка) ---
st.sidebar.header("📂 Данные")

# КНОПКА ЗАГРУЗКИ (Upload CSV)
uploaded_file = st.sidebar.file_uploader("Загрузите отчет LiveSklad (CSV)", type="csv")

if uploaded_file is not None:
    # Если Алексей скинул файл и мы его загрузили
    df = pd.read_csv(uploaded_file)
    df['Дата'] = pd.to_datetime(df['Дата'], dayfirst=True)
    df['Месяц'] = df['Дата'].dt.to_period('M').astype(str)
else:
    # Иначе используем наш фейковый файл по умолчанию
    df = load_data()

if df is None:
    st.error("❌ Файл данных не найден! Загрузите файл или запустите генератор.")
    st.stop()

st.sidebar.header("⚙️ Настройки фильтрации")

# --- ДОБАВЛЯЕМ ПЕРЕКЛЮЧАТЕЛЬ ЦЕЛИ ПРОГНОЗА ---
st.sidebar.divider()
target_type = st.sidebar.radio("🎯 Что прогнозировать?", ["Выручка (₽)", "Количество (шт)"])

# Определяем, какую колонку будем считать
if target_type == "Выручка (₽)":
    target_col = "Сумма"
else:
    target_col = "Кол-во"

# Функция классификации
def classify(row):
    if row['Тип документа'] == 'Продажа': return 'Товар'
    return 'Услуга'
df['Категория'] = df.apply(classify, axis=1)

# Фильтр категорий
selected_type = st.sidebar.multiselect(
    "1. Выберите категорию:", 
    options=["Товар", "Услуга"],
    default=["Товар", "Услуга"]
)
df_filtered = df[df['Категория'].isin(selected_type)]

# ВЫПАДАЮЩИЙ СПИСОК ТОВАРОВ (Если их много)
unique_items = df_filtered['Название'].unique()
selected_items = st.sidebar.multiselect(
    "2. Выберите конкретные позиции (опционально):", 
    options=unique_items,
    default=unique_items
)
df_filtered = df_filtered[df_filtered['Название'].isin(selected_items)]

# Функция классификации
def classify(row):
    if row['Тип документа'] == 'Продажа': return 'Товар'
    return 'Услуга'

df['Категория'] = df.apply(classify, axis=1)

# Фильтр
selected_type = st.sidebar.multiselect(
    "Выберите категорию:", 
    options=["Товар", "Услуга"],
    default=["Товар", "Услуга"]
)

df_filtered = df[df['Категория'].isin(selected_type)]

# KPI
st.subheader("💰 Финансовые показатели")

total_revenue = df_filtered['Сумма'].sum()
total_profit = df_filtered['Валовая прибыль'].sum()

# Расчет чистой прибыли
service_revenue = df_filtered[df_filtered['Категория'] == 'Услуга']['Сумма'].sum()
master_salary = service_revenue * 0.4
net_profit = total_profit - master_salary

col1, col2, col3 = st.columns(3)
col1.metric("Оборот (Выручка)", f"{total_revenue:,.0f} ₽")
col2.metric("Валовая прибыль", f"{total_profit:,.0f} ₽")
col3.metric("Чистая прибыль (Расчетная)", f"{net_profit:,.0f} ₽", help="Валовая - 40% от услуг")

# Графики
st.divider()

col_left, col_right = st.columns(2)

with col_left:
    st.subheader("Динамика выручки по месяцам")
    
    monthly_sales = df_filtered.groupby('Месяц')['Сумма'].sum().reset_index()
    
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    
    sns.barplot(data=monthly_sales, x='Месяц', y='Сумма', ax=ax1, color="#4b8bbe")
    
    # Настройки осей
    ax1.set_xlabel("Месяц")
    ax1.set_ylabel("Выручка (руб)")
    plt.xticks(rotation=45)
    
    st.pyplot(fig1)

with col_right:
    st.subheader("Топ-5 Популярных позиций")
    
    top_items = df_filtered.groupby('Название')['Сумма'].sum().nlargest(5)
    
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    
    ax2.pie(top_items, labels=top_items.index, autopct='%1.1f%%', startangle=140, colors=sns.color_palette("pastel"))
    ax2.set_title("Доля в выручке")
    
    st.pyplot(fig2)

# Таблица данных
st.divider()
st.subheader("📋 Детальный отчет")

with st.expander("Открыть таблицу данных"):
    st.dataframe(df_filtered, use_container_width=True)
    
    csv = df_filtered.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Скачать таблицу в CSV",
        data=csv,
        file_name="filtered_report.csv",
        mime="text/csv",
    )

# --- 5.5 Сравнение Год к Году (YoY - Year over Year) ---
st.divider()
st.subheader("📊 Сравнение продаж: Год к Году")

# Чтобы не было предупреждений от Pandas, делаем копию
df_yoy = df_filtered.copy()
df_yoy['Год'] = df_yoy['Дата'].dt.year
df_yoy['Номер_месяца'] = df_yoy['Дата'].dt.month

# Группируем данные
df_yoy_grouped = df_yoy.groupby(['Год', 'Номер_месяца'])['Сумма'].sum().reset_index()

fig_yoy, ax_yoy = plt.subplots(figsize=(10, 4))
# Рисуем график, где цветом (hue) выделен год
sns.barplot(data=df_yoy_grouped, x='Номер_месяца', y='Сумма', hue='Год', palette='viridis', ax=ax_yoy)

ax_yoy.set_xlabel("Месяц (1 - Январь, 12 - Декабрь)")
ax_yoy.set_ylabel("Выручка (руб)")
ax_yoy.set_title("Сравнение выручки по годам")
st.pyplot(fig_yoy)

# Машинное обучение
st.divider()
df_monthly = df_filtered.groupby('Месяц', as_index=False)[target_col].sum()

if len(df_monthly) < 3:
    st.warning("⚠️ Недостаточно данных для прогноза. Нужно минимум 3 месяца.")
else:
    df_monthly['Month_ID'] = range(len(df_monthly))
    X = df_monthly[['Month_ID']].values 
    y = df_monthly[target_col].values # <--- Тут теперь могут быть и Штуки, и Рубли!

    # Обучаем модель
    from sklearn.linear_model import LinearRegression
    model = LinearRegression()
    model.fit(X, y) 

    # --- ОЦЕНКА ТОЧНОСТИ МОДЕЛИ (ДЛЯ ДИПЛОМА) ---
    y_pred_train = model.predict(X) # Просим модель предсказать прошлое, чтобы сравнить с фактом
    r2 = r2_score(y, y_pred_train)  # Коэффициент детерминации (в %)
    mae = mean_absolute_error(y, y_pred_train) # Средняя ошибка
    
    # Защита от отрицательного R2 (если данные слишком хаотичные)
    r2_display = max(0, r2) * 100 

    st.markdown("### 📊 Метрики качества модели")
    col_m1, col_m2 = st.columns(2)
    col_m1.metric("Точность тренда (R²)", f"{r2_display:.1f}%", help="Ближе к 100% = отличный прогноз")
    col_m2.metric("Средняя ошибка (MAE)", f"{mae:.1f}", help=f"На столько {target_type} мы ошибаемся в среднем")

    # Прогноз в будущее
    future_months = st.slider("🗓 На сколько месяцев вперед сделать прогноз?", min_value=1, max_value=12, value=6)
    last_id = df_monthly['Month_ID'].max() 
    future_X = np.arange(last_id + 1, last_id + 1 + future_months).reshape(-1, 1)
    future_pred = model.predict(future_X)

    fig3, ax3 = plt.subplots(figsize=(8, 4))

    # Рисуем реальную историю
    sns.lineplot(x=df_monthly['Month_ID'], y=y, ax=ax3, label="Факт (История)", marker='o', color='black')

    # Рисуем прогноз
    plt.plot(future_X, future_pred, label="Прогноз (ML)", color='red', linestyle='--', marker='x')

    ax3.set_title("Прогноз выручки на полгода")
    ax3.set_xlabel("Номер месяца (с начала наблюдений)")
    ax3.set_ylabel("Выручка (руб)")
    ax3.legend()
    ax3.grid(True)

    st.pyplot(fig3)

    # --- 8. Умный аналитик (Система поддержки принятия решений) ---
    st.divider()
    st.subheader("🧠 Аналитическая сводка и бизнес-рекомендации")

    # Проверка на количество данных (нужно хотя бы 3 месяца по ТЗ)
    if len(df_monthly) < 3:
        st.warning("⚠️ Недостаточно исторических данных для точного прогноза. Для корректного анализа загрузите файл с историей минимум за 3 месяца.")
    else:
        # Шаг 1: Расчет тренда
        fact = y[-1]  # Последний реальный месяц продаж
        forecast = future_pred[0]  # Прогноз на следующий ближайший месяц
        
        # Защита от ошибки деления на ноль (если вдруг продаж было 0)
        if fact == 0:
            fact = 1 
            
        # Считаем разницу в процентах
        delta = ((forecast - fact) / fact) * 100

        # Шаг 2: Генерация текста (Правила)
        if delta < -15:
            st.error(f"🔴 **Внимание: Прогнозируется спад спроса!**\n\n"
                     f"Ожидается снижение показателей на **{abs(delta):.1f}%**. Вероятно сезонное затишье.\n\n"
                     f"**Рекомендации:**\n"
                     f"1. Сократить закупку запчастей во избежание затоваривания склада.\n"
                     f"2. Рассмотреть запуск рекламной акции или скидок на услуги ремонта для привлечения клиентов.")
                     
        elif delta > 15:
            st.success(f"🟢 **Позитивный тренд: Рост спроса.**\n\n"
                       f"Прогнозируется увеличение спроса на **{delta:.1f}%** по сравнению с прошлым месяцем.\n\n"
                       f"**Рекомендации:**\n"
                       f"1. Увеличить складские запасы расходных материалов.\n"
                       f"2. Проверить график смен мастеров, возможно потребуется усиление штата для предотвращения очередей.")
                       
        else:
            # Знак плюс выводится автоматически благодаря форматированию {:+.1f}
            st.info(f"🟡 **Стабильная ситуация.**\n\n"
                    f"Значимых колебаний спроса не предвидится (прогнозируемое изменение: **{delta:+.1f}%**).\n\n"
                    f"**Рекомендации:**\n"
                    f"Поддерживать текущий уровень закупок и работы персонала.")