import streamlit as st
import pandas as pd
import seaborn as sns
import numpy as np

from modules.data_processor import load_data, classify_smart
from modules.charts import draw_revenue_bar, draw_top_items_pie, draw_yoy_chart, draw_forecast_chart
from modules.ml_model import run_prediction

# --- Настройки ---
st.set_page_config(page_title="Аналитика Сервиса", layout="wide")
st.title("📊 Система прогнозирования и аналитики")
sns.set_theme(style="whitegrid")

# --- 1. ЗАГРУЗКА ---
st.sidebar.header("📂 Данные")
uploaded_file = st.sidebar.file_uploader("Загрузите отчет LiveSklad", type=["csv", "xlsx"])

if uploaded_file is not None:
    df = load_data(uploaded_file)
else:
    df = load_data("data/livesklad_export.csv")

if df is None:
    st.error("❌ Файл данных не найден!")
    st.stop()

# --- 2. ФИЛЬТРЫ ---
st.sidebar.header("⚙️ Настройки фильтрации")
target_type = st.sidebar.radio(
    "🎯 Что прогнозировать?",
    ["Выручка (₽)", "Количество (шт)"], 
    help="Выберите, в каких единицах строить график: в деньгах (для оценки бюджета) или в штуках (для планирования закупок на склад)."
)

# Берем правильное название колонки!
target_col = "Сумма" if target_type == "Выручка (₽)" else "Количество"

st.sidebar.divider()
use_smart_sort = st.sidebar.checkbox(
    "Включить умную группировку (Специфика СЦ)", 
    value=True, 
    help="Если включено: программа сама найдет в чеках слова 'Дисплей', 'Чехол' и распределит их по категориям (Запчасти, Аксессуары). Если выключено: оставит как есть в отчете."
)

# Применяем классификацию
df['Категория'] = df.apply(lambda row: classify_smart(row, use_smart_sort), axis=1)

unique_categories = df['Категория'].unique()
selected_type = st.sidebar.multiselect(
    "1. Выберите категорию:", 
    options=unique_categories, 
    default=unique_categories,
    help="Оставьте нужные бизнес-направления. Графики перестроятся автоматически."
)
df_filtered = df[df['Категория'].isin(selected_type)]

# --- ФИЛЬТР ТОВАРОВ ---
unique_items = df_filtered['Название'].unique()
selected_items = st.sidebar.multiselect(
    "2. Выберите конкретные позиции:", 
    options=unique_items,
    default=[],
    help="Оставьте поле пустым, чтобы анализировать всю категорию целиком. Выберите 1-2 товара, чтобы посмотреть прогноз только по ним."
)

# Магия: если список не пустой, фильтруем. Если пустой - оставляем всё!
if len(selected_items) > 0:
    df_filtered = df_filtered[df_filtered['Название'].isin(selected_items)]

# --- 3. ЭКОНОМИКА ---
st.sidebar.divider()
st.sidebar.header("💰 Экономика бизнеса")
master_percent = st.sidebar.slider(
    "Процент ЗП мастера с услуг (%)", 
    min_value=0, max_value=100, value=40,
    help="Укажите процент сдельной оплаты мастеров. Эта сумма будет вычитаться из выручки за 'Услуги' при расчете чистой прибыли."
)

# --- 4. ИНТЕРФЕЙС (ОТРИСОВКА) ---
st.subheader("💰 Финансовые показатели")
show_net_profit = st.toggle("Показать расчетную чистую прибыль (с вычетом ЗП мастеров)")

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

# --- СОВЕТНИК V2 ---
    st.divider()
    st.subheader("🧠 Аналитическая сводка и бизнес-рекомендации")
    
    # 1. Оцениваем надежность прогноза (по метрике R2)
    reliability = "Высокая 🟢" if r2 > 60 else "Средняя 🟡" if r2 > 30 else "Низкая (рынок нестабилен) 🔴"

    # 2. Считаем тренд (сравниваем среднее будущее со средним за последние 3 месяца)
    avg_fact = np.mean(y[-3:]) if len(y) >= 3 else np.mean(y)
    avg_forecast = np.mean(future_pred)
    
    if avg_fact == 0: avg_fact = 1 # Защита от деления на ноль
    delta = ((avg_forecast - avg_fact) / avg_fact) * 100

    # 3. Определяем контекст
    is_services = any(cat in selected_type for cat in ["Услуги", "Услуга"])
    is_goods = any(cat in selected_type for cat in ["Запчасти", "Аксессуары", "Товар"])

    # 4. Формируем УМНЫЙ вывод
    if delta < -10:
        st.error(f"📉 **Тренд: Ожидается СПАД на {abs(delta):.1f}%**")
        st.write(f"**Надежность прогноза:** {reliability}")
        st.write("**📝 План действий:**")
        if is_goods:
            st.write("📦 **По складу:** Сократите закупки запчастей и аксессуаров. Проведите инвентаризацию и устройте распродажу залежавшегося товара (неликвида), чтобы высвободить замороженные деньги.")
        if is_services:
            st.write("👨‍🔧 **По ремонтам:** Ожидается просадка по заказам. Запустите стимулирующие акции (например, 'Бесплатная чистка динамиков при замене экрана') для привлечения трафика.")
            
    elif delta > 10:
        st.success(f"📈 **Тренд: Ожидается РОСТ на {delta:.1f}%**")
        st.write(f"**Надежность прогноза:** {reliability}")
        st.write("**📝 План действий:**")
        if is_goods:
            st.write("📦 **По складу:** Сформируйте запас ходовых позиций заранее. Свяжитесь с поставщиками для резервирования партий, чтобы избежать пустых полок и отказов клиентам.")
        if is_services:
            st.write("👨‍🔧 **По ремонтам:** Возможен наплыв клиентов. Проверьте графики отпусков мастеров. Возможно, потребуется вывести дополнительных сотрудников в смены, чтобы избежать очередей.")
            
    else:
        st.info(f"⚖️ **Тренд: СТАБИЛЬНЫЙ (изменение {delta:+.1f}%)**")
        st.write(f"**Надежность прогноза:** {reliability}")
        st.write("**📝 План действий:**")
        st.write("Работаем в штатном режиме. Поддерживайте стандартный 'несгораемый остаток' на складе и текущий график работы мастеров.")
        
    if r2 < 30:
        st.caption("⚠️ *Примечание ИИ: Исторические данные имеют сильные хаотичные скачки. Математической модели сложно выявить четкий тренд. Рекомендуется принимать решения с осторожностью и осуществлять закупки мелкими партиями.*")