import streamlit as st
import pandas as pd
import seaborn as sns

# Импортируем наши собственные модули!
from modules.data_processor import load_data, classify_smart
from modules.charts import draw_revenue_bar, draw_top_items_pie, draw_yoy_chart, draw_forecast_chart
from modules.ml_model import run_prediction

# --- Настройки ---
st.set_page_config(page_title="Аналитика Сервиса", layout="wide")
st.title("📊 Система прогнозирования и аналитики")
sns.set_theme(style="whitegrid")

# --- 1. ЗАГРУЗКА ---
st.sidebar.header("📂 Данные")
uploaded_file = st.sidebar.file_uploader("Загрузите отчет LiveSklad (CSV)", type="csv")

if uploaded_file is not None:
    df = load_data(uploaded_file)
else:
    df = load_data("data/livesklad_export.csv")

if df is None:
    st.error("❌ Файл данных не найден!")
    st.stop()

# --- 2. ФИЛЬТРЫ ---
st.sidebar.header("⚙️ Настройки фильтрации")
target_type = st.sidebar.radio("🎯 Что прогнозировать?",["Выручка (₽)", "Количество (шт)"])
target_col = "Сумма" if target_type == "Выручка (₽)" else "Кол-во"

st.sidebar.divider()
use_smart_sort = st.sidebar.checkbox("Включить умную группировку (Специфика СЦ)", value=True)

# Применяем классификацию
df['Категория'] = df.apply(lambda row: classify_smart(row, use_smart_sort), axis=1)

unique_categories = df['Категория'].unique()
selected_type = st.sidebar.multiselect("1. Выберите категорию:", unique_categories, default=unique_categories)
df_filtered = df[df['Категория'].isin(selected_type)]

unique_items = df_filtered['Название'].unique()
selected_items = st.sidebar.multiselect("2. Выберите конкретные позиции:", unique_items, default=unique_items)
df_filtered = df_filtered[df_filtered['Название'].isin(selected_items)]

# --- 3. ЭКОНОМИКА ---
st.sidebar.divider()
st.sidebar.header("💰 Экономика бизнеса")
master_percent = st.sidebar.slider("Процент ЗП мастера с услуг (%)", 0, 100, 40)

# --- 4. ИНТЕРФЕЙС (ОТРИСОВКА) ---
st.subheader("💰 Финансовые показатели")
show_net_profit = st.toggle("Показать расчетную чистую прибыль (с вычетом ЗП мастеров)")

total_revenue = df_filtered['Сумма'].sum()
total_gross_profit = df_filtered['Валовая прибыль'].sum()

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

# --- ГРАФИКИ ---
st.divider()
col_left, col_right = st.columns(2)
with col_left:
    st.subheader("Динамика выручки по месяцам")
    st.pyplot(draw_revenue_bar(df_filtered)) # Вызов из файла charts.py
with col_right:
    st.subheader("Топ-10 Популярных позиций")
    st.pyplot(draw_top_items_pie(df_filtered)) # Вызов из файла charts.py

# --- ТАБЛИЦА ---
st.divider()
st.subheader("📋 Детальный отчет")
with st.expander("Открыть таблицу данных"):
    st.dataframe(df_filtered, use_container_width=True)
    st.download_button("📥 Скачать таблицу в CSV", df_filtered.to_csv(index=False).encode('utf-8'), "filtered_report.csv", "text/csv")

# --- YOY СРАВНЕНИЕ ---
st.divider()
st.subheader("📊 Сравнение продаж: Год к Году")
st.pyplot(draw_yoy_chart(df_filtered)) # Вызов из файла charts.py

# --- МАШИННОЕ ОБУЧЕНИЕ ---
st.divider()
st.subheader("🔮 Прогноз спроса (Linear Regression)")
df_monthly = df_filtered.groupby('Месяц', as_index=False)[target_col].sum()

if len(df_monthly) < 3:
    st.warning("⚠️ Недостаточно данных для прогноза. Нужно минимум 3 месяца.")
else:
    future_months = st.slider("🗓 На сколько месяцев вперед сделать прогноз?", 1, 12, 6)
    
    # Вся сложная математика спрятана внутри этой одной строки!
    X, y, future_X, future_pred, r2, mae = run_prediction(df_monthly, target_col, future_months)

    st.markdown("### 📊 Метрики качества модели")
    col_m1, col_m2 = st.columns(2)
    col_m1.metric("Точность тренда (R²)", f"{r2:.1f}%", help="Ближе к 100% = отличный прогноз")
    col_m2.metric("Средняя ошибка (MAE)", f"{mae:.1f}", help=f"На столько {target_type} мы ошибаемся в среднем")

    st.pyplot(draw_forecast_chart(df_monthly, y, future_X, future_pred, target_type))

    # --- СОВЕТНИК ---
    st.divider()
    st.subheader("🧠 Аналитическая сводка и бизнес-рекомендации")
    fact = y[-1] if y[-1] != 0 else 1 
    forecast = future_pred[0]
    delta = ((forecast - fact) / fact) * 100

    if delta < -15:
        st.error(f"🔴 **Внимание: Прогнозируется спад спроса!**\n\nОжидается снижение на **{abs(delta):.1f}%**.\n\n**Рекомендации:**\n1. Сократить закупку.\n2. Запустить акцию.")
    elif delta > 15:
        st.success(f"🟢 **Позитивный тренд: Рост спроса.**\n\nПрогнозируется увеличение на **{delta:.1f}%**.\n\n**Рекомендации:**\n1. Увеличить запасы.\n2. Проверить штат мастеров.")
    else:
        st.info(f"🟡 **Стабильная ситуация.**\n\nИзменение: **{delta:+.1f}%**.\n\n**Рекомендации:**\nПоддерживать текущий уровень закупок.")