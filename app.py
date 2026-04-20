import streamlit as st
import pandas as pd
import seaborn as sns
import numpy as np

# Обновляем импорты: добавляем новые функции из charts.py
from modules.data_processor import load_data, classify_smart
from modules.charts import (
    draw_revenue_bar, 
    draw_top_items_pie, 
    draw_yoy_chart, 
    draw_forecast_chart, 
    perform_abc_analysis, 
    draw_seasonality_chart
)
from modules.ml_model import run_prediction

# --- Настройки ---
st.set_page_config(page_title="Прогнозирование спроса", layout="wide", page_icon="📊", initial_sidebar_state="expanded")
st.title("📊 Система прогнозирования и аналитики")
sns.set_theme(style="darkgrid")

# --- 1. ЗАГРУЗКА ---
st.sidebar.header("📂 Данные")
uploaded_file = st.sidebar.file_uploader("Загрузите отчет LiveSklad", type=["csv", "xlsx"])

if uploaded_file is not None:
    df = load_data(uploaded_file)
else:
    # Убедись, что тестовый файл существует по этому пути
    try:
        df = load_data("data/Отчет по товарам и работам.xlsx")
    except FileNotFoundError:
        st.sidebar.error("Тестовый файл не найден. Пожалуйста, загрузите свой отчет.")
        st.stop()


if df is None:
    st.error("❌ Файл данных не может быть загружен. Проверьте формат файла.")
    st.stop()

# --- 2. ФИЛЬТРЫ ---
st.sidebar.header("⚙️ Настройки фильтрации")
target_type = st.sidebar.radio(
    "🎯 Что анализировать?",
    ["Выручка (₽)", "Количество (шт)"], 
    help="Выберите, в каких единицах строить графики: в деньгах (для оценки бюджета) или в штуках (для планирования закупок)."
)

target_col = "Сумма" if target_type == "Выручка (₽)" else "Количество"

st.sidebar.divider()
use_smart_sort = st.sidebar.checkbox(
    "Включить умную группировку", 
    value=True, 
    help="Программа сама найдет в чеках 'Дисплей', 'Чехол' и распределит их по категориям. Если выключено - оставит как в отчете."
)

df['Категория'] = df.apply(lambda row: classify_smart(row, use_smart_sort), axis=1)
unique_categories = sorted(df['Категория'].unique())
selected_type = st.sidebar.multiselect(
    "1. Выберите категорию:", 
    options=unique_categories, 
    default=unique_categories,
    help="Оставьте нужные бизнес-направления. Графики перестроятся."
)

if not selected_type:
    st.warning("Пожалуйста, выберите хотя бы одну категорию.")
    st.stop()
    
df_filtered = df[df['Категория'].isin(selected_type)]

unique_items = sorted(df_filtered['Название'].unique())
selected_items = st.sidebar.multiselect(
    "2. Выберите конкретные позиции:", 
    options=unique_items,
    default=[],
    help="Оставьте поле пустым, чтобы анализировать всю категорию. Выберите 1-2 товара, чтобы посмотреть прогноз только по ним."
)

if len(selected_items) > 0:
    df_filtered = df_filtered[df_filtered['Название'].isin(selected_items)]

if df_filtered.empty:
    st.warning("По выбранным фильтрам нет данных. Попробуйте изменить выбор.")
    st.stop()

# --- 3. ЭКОНОМИКА ---
st.sidebar.divider()
st.sidebar.header("💰 Экономика бизнеса")
master_percent = st.sidebar.slider(
    "Процент ЗП мастера с услуг (%)", 
    min_value=0, max_value=100, value=40,
    help="Укажите процент сдельной оплаты мастеров для расчета чистой прибыли."
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

st.divider()

# --- ВКЛАДКИ ДЛЯ ОРГАНИЗАЦИИ КОНТЕНТА ---
tab1, tab2, tab3 = st.tabs(["📊 Общий обзор", "📈 Детальный анализ", "🔮 Прогнозирование"])

# --- ВКЛАДКА 1: ОБЩИЙ ОБЗОР ---
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

# --- ВКЛАДКА 2: ДЕТАЛЬНЫЙ АНАЛИЗ ---
with tab2:
    st.subheader("📋 Детальный отчет")
    with st.expander("Открыть таблицу с данными"):
        st.dataframe(df_filtered, width='stretch')
        st.download_button("📥 Скачать таблицу в CSV", df_filtered.to_csv(index=False, encoding='utf-8-sig'), "filtered_report.csv", "text/csv")
    
    st.divider()
    st.subheader("📦 ABC-анализ товаров", help="Делит товары на 3 группы: A - самые прибыльные (80% выручки), B - стабильные (15%), C - незначительные (5%).")
    abc_df = perform_abc_analysis(df_filtered)
    
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.success("Группа A (Ключевые)")
        st.dataframe(abc_df[abc_df['ABC Категория'] == 'A (Ключевые)'], height=300)
    with col_b:
        st.warning("Группа B (Стабильные)")
        st.dataframe(abc_df[abc_df['ABC Категория'] == 'B (Стабильные)'], height=300)
    with col_c:
        st.error("Группа C (Незначительные)")
        st.dataframe(abc_df[abc_df['ABC Категория'] == 'C (Незначительные)'], height=300)
        
    st.divider()
    st.subheader("📅 Анализ сезонности", help="Показывает, в какие месяцы продажи обычно выше или ниже среднего.")
    st.pyplot(draw_seasonality_chart(df_filtered, target_col))

# --- ВКЛАДКА 3: ПРОГНОЗИРОВАНИЕ ---
with tab3:
    st.subheader("🔮 Прогноз спроса (Linear Regression)")
    df_monthly = df_filtered.groupby('Месяц', as_index=False)[target_col].sum()

    if len(df_monthly) < 3:
        st.warning("⚠️ Недостаточно данных для прогноза. Нужно минимум 3 месяца.")
    else:
        future_months = st.slider("🗓 На сколько месяцев вперед сделать прогноз?", 1, 12, 6)
        
        X, y_values, future_X, future_pred, r2, mae = run_prediction(df_monthly, target_col, future_months)

        st.pyplot(draw_forecast_chart(df_monthly, df_monthly[target_col], future_X, future_pred, target_type))

        st.markdown("##### 📊 Метрики качества модели")
        col_m1, col_m2 = st.columns(2)
        col_m1.metric("Точность тренда (R²)", f"{r2:.1f}%", help="Насколько хорошо линия тренда описывает исторические данные. Ближе к 100% = лучше.")
        col_m2.metric("Средняя ошибка (MAE)", f"{mae:,.0f}", help=f"В среднем, прогноз ошибается на это количество {target_type.lower()}.")
        
        st.divider()
        st.subheader("🧠 Аналитическая сводка и бизнес-рекомендации")
        
        avg_fact = np.mean(y_values[-3:]) if len(y_values) >= 3 else np.mean(y_values)
        avg_forecast = np.mean(future_pred)
        if avg_fact == 0: avg_fact = 1 
        delta = ((avg_forecast - avg_fact) / avg_fact) * 100

        types_to_check = selected_type if len(selected_type) > 0 else unique_categories
        is_services = any(cat in types_to_check for cat in ["Услуги", "Услуга"])
        is_goods = any(cat in types_to_check for cat in ["Запчасти", "Аксессуары", "Товар", "Прочее"])
        
        if delta > 10:
            st.success(f"📈 **Тренд: Уверенный РОСТ на ~{delta:.0f}%**")
            st.write("**📝 План действий:**")
            if target_col == "Количество":
                if is_goods: st.write("📦 **Склад:** Увеличьте объем закупок. Создайте страховой запас (+15-20%).")
                if is_services: st.write("👨‍🔧 **Персонал:** Прогнозируется рост потока клиентов. Проверьте график мастеров.")
            else:
                st.write("💰 **Бюджет:** Ожидается рост денежного потока. Благоприятный период для инвестиций.")
        elif delta < -10:
            st.error(f"📉 **Тренд: Прогнозируется СПАД на ~{abs(delta):.0f}%**")
            st.write("**📝 План действий:**")
            if target_col == "Количество":
                if is_goods: st.write("📦 **Склад:** Приостановите закупки. Распродавайте остатки.")
                if is_services: st.write("👨‍🔧 **Персонал:** Загрузка мастеров может упасть. Запустите акции/скидки.")
            else:
                st.write("💰 **Бюджет:** Ожидается снижение выручки. Оптимизируйте расходы.")
        else:
            st.info(f"⚖️ **Тренд: СТАБИЛЬНЫЙ (изменение {delta:+.1f}%)**")
            st.write("**📝 План действий:**")
            if target_col == "Количество":
                st.write("Поддерживайте стандартный уровень запасов.")
            else:
                st.write("Финансовые показатели стабильны. Придерживайтесь текущего плана.")
            
        if r2 < 30:
            st.caption("⚠️ *Примечание: Исторические данные нестабильны. Модели сложно выявить четкий тренд. Рекомендуется принимать решения с осторожностью.*")