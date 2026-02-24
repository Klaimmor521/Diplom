import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
import numpy as np

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

# Фильтры
st.sidebar.header("Настройки фильтрации")

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

# Таблица данныых
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

# Машинное обучение
st.divider()
st.subheader("🔮 Прогноз спроса (Linear Regression)")

df_monthly = df_filtered.groupby('Месяц', as_index=False)['Сумма'].sum()

if len(df_monthly) < 2:
    st.warning("⚠️ Недостаточно данных для прогноза. Нужно минимум 2 месяца продаж.")
else:
    # Превращаем даты в цифры
    df_monthly['Month_ID'] = range(len(df_monthly))

    # Выделяем X (Время) и Y (Продажи)
    X = df_monthly[['Month_ID']].values 
    y = df_monthly['Сумма'].values

    # Создаем и обучаем модель
    model = LinearRegression()
    model.fit(X, y)

    # Прогноз в будущее
    future_months = 6
    last_id = df_monthly['Month_ID'].max()
    
    future_X = np.arange(last_id + 1, last_id + 1 + future_months).reshape(-1, 1)
    
    future_pred = model.predict(future_X)

    fig3, ax3 = plt.subplots(figsize=(8, 4))

    # Рисуем реальную историю
    sns.lineplot(x=df_monthly['Month_ID'], y=y, ax=ax3, label="Факт (История)", marker='o', color='blue')

    # Рисуем прогноз
    plt.plot(future_X, future_pred, label="Прогноз (ML)", color='red', linestyle='--', marker='x')

    ax3.set_title("Прогноз выручки на полгода")
    ax3.set_xlabel("Номер месяца (с начала наблюдений)")
    ax3.set_ylabel("Выручка (руб)")
    ax3.legend()
    ax3.grid(True)

    st.pyplot(fig3)

    # Умный совет
    last_real = y[-1]
    last_pred = future_pred[-1]
    
    change = ((last_pred - last_real) / last_real) * 100

    st.subheader("📢 Анализ тренда:")
    if change > 5:
        st.success(f"📈 Ожидается РОСТ спроса на **{change:.1f}%**. Рекомендуется увеличить закупку!")
    elif change < -5:
        st.error(f"📉 Ожидается СПАД на **{abs(change):.1f}%**. Оптимизируйте склад, не закупайте лишнего.")
    else:
        st.info(f"➡️ Спрос стабилен (изменение **{change:.1f}%**). Работаем в штатном режиме.")