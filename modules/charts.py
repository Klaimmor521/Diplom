import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

sns.set_theme(style="whitegrid", palette="colorblind")

# --- 1. Гистограмма выручки по месяцам ---
def draw_revenue_bar(df_filtered, target_col):
    monthly_sales = df_filtered.groupby('Месяц')[target_col].sum().reset_index()
    fig, ax = plt.subplots(figsize=(8, 6))
    
    barplot = sns.barplot(data=monthly_sales, x='Месяц', y=target_col, ax=ax, color="#5b9bd5", edgecolor="none", alpha=0.9)
    
    for p in barplot.patches:
        ax.annotate(format(p.get_height(), '.0f'), 
                    (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha = 'center', va = 'center', 
                    xytext = (0, 9), 
                    textcoords = 'offset points',
                    fontsize=8, color='#333333', rotation=45)
    
    ax.set_xlabel("")
    
    # Меняем подпись сбоку
    ylabel_text = "Выручка (руб)" if target_col == "Сумма" else "Количество (шт)"
    ax.set_ylabel(ylabel_text)
    
    plt.xticks(rotation=45)
    sns.despine(left=True)
    return fig

# --- 2. Круговая диаграмма (Топ-10) ---
def draw_top_items_pie(df_filtered, target_col):
    # Группируем по выбранной колонке
    top_items = df_filtered.groupby('Название')[target_col].sum().nlargest(10)
    fig, ax = plt.subplots(figsize=(10, 6))
    
    wedges, texts, autotexts = ax.pie(
        top_items, 
        labels=None,
        autopct='%1.1f%%', 
        startangle=140, 
        colors=sns.color_palette("pastel"), 
        wedgeprops=dict(width=0.4, edgecolor='w'),
        textprops=dict(color="black", fontsize=9)
    )
    
    # Умный заголовок в зависимости от того, что считаем
    title_text = "Доля в выручке (₽)" if target_col == "Сумма" else "Доля в продажах (шт)"
    ax.set_title(title_text, pad=20)
    
    ax.legend(
        wedges, 
        top_items.index, 
        title="Названия позиций", 
        loc="center left", 
        bbox_to_anchor=(1, 0, 0.5, 1)
    )
    
    return fig

# --- 3. Сравнение год к году (YoY) ---
def draw_yoy_chart(df_filtered, target_col):
    df_yoy = df_filtered.copy()
    df_yoy['Год'] = df_yoy['Дата'].dt.year
    df_yoy['Номер_месяца'] = df_yoy['Дата'].dt.month
    
    # Группируем по нужной колонке
    df_yoy_grouped = df_yoy.groupby(['Год', 'Номер_месяца'])[target_col].sum().reset_index()
    
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(data=df_yoy_grouped, x='Номер_месяца', y=target_col, hue='Год', palette='Set2', ax=ax, edgecolor="none", alpha=0.85)
    
    ax.set_xlabel("Месяц (1 - Январь, 12 - Декабрь)")
    
    # Меняем заголовок и ось
    ylabel_text = "Выручка (руб)" if target_col == "Сумма" else "Количество (шт)"
    title_text = "Сравнение выручки по годам" if target_col == "Сумма" else "Сравнение продаж (в штуках) по годам"
    
    ax.set_ylabel(ylabel_text)
    ax.set_title(title_text, pad=15)
    
    sns.despine(left=True)
    return fig

# --- 4. Прогноз с заливкой (Area Chart) ---
def draw_forecast_chart(df_monthly, y, future_X, future_pred, target_type):
    fig, ax = plt.subplots(figsize=(8, 5))

    moving_average = df_monthly[y.name].rolling(window=3, center=True, min_periods=1).mean()
    sns.lineplot(x=df_monthly['Month_ID'], y=moving_average, ax=ax, label="Скользящее среднее (Тренд)", color='#ff7f0e', linestyle=':', linewidth=2.5)

    sns.lineplot(x=df_monthly['Month_ID'], y=y, ax=ax, label="Факт (История)", marker='o', color='#286090', linewidth=2.5, alpha=0.7)
    ax.plot(future_X, future_pred, label="Прогноз (ML)", color='#d62728', linestyle='--', marker='s', linewidth=2)

    last_x_hist = df_monthly['Month_ID'].iloc[-1]
    last_y_hist = y.iloc[-1]
    ax.annotate(f'{last_y_hist:,.0f}', xy=(last_x_hist, last_y_hist), xytext=(-15, 15), textcoords='offset points', weight='bold', color='#286090')

    last_x_pred = future_X[-1][0]
    last_y_pred = future_pred[-1]
    ax.annotate(f'{last_y_pred:,.0f}', xy=(last_x_pred, last_y_pred), xytext=(-15, 15), textcoords='offset points', weight='bold', color='#d62728')

    ax.set_title(f"Прогноз: {target_type}", pad=15)
    ax.set_xlabel("Номер месяца")
    ax.set_ylabel(target_type)
    ax.legend(loc="upper left")

    import matplotlib.ticker as ticker
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    sns.despine() 
    return fig

# --- 5. ABC ---
def perform_abc_analysis(df_filtered):
    """
    Выполняет ABC-анализ номенклатуры по выручке.
    Возвращает датафрейм с категориями A, B, C и колонкой для прогресс-бара.
    """
    if 'Сумма' not in df_filtered.columns or df_filtered['Сумма'].sum() == 0:
        return pd.DataFrame()

    item_revenue = df_filtered.groupby('Название')['Сумма'].sum().sort_values(ascending=False).reset_index()
    item_revenue.rename(columns={'Сумма': 'Выручка'}, inplace=True)
    
    total_revenue = item_revenue['Выручка'].sum()
    item_revenue['Доля в выручке'] = (item_revenue['Выручка'] / total_revenue)
    item_revenue['Накопительная доля'] = item_revenue['Доля в выручке'].cumsum()
    
    # --- НОВОЕ: Создаем колонку с процентами для ProgressColumn ---
    # Мы умножаем долю на 100, чтобы получить удобное число для бара (например, 15.25)
    item_revenue['Доля в %'] = item_revenue['Доля в выручке'] * 100
    
    def assign_abc_category(share):
        if share <= 0.8: return 'A'
        if share <= 0.95: return 'B'
        return 'C'
    
    item_revenue['Категория'] = item_revenue['Накопительная доля'].apply(assign_abc_category)
    
    # Выбираем нужные колонки для итоговой таблицы
    result_df = item_revenue[['Категория', 'Название', 'Выручка', 'Доля в %']]
    return result_df

# --- 6. Box Plot ---
def draw_seasonality_chart(df_filtered, target_col):
    df_season = df_filtered.copy()
    df_season['Номер_месяца'] = df_season['Дата'].dt.month
    
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.boxplot(data=df_season, x='Номер_месяца', y=target_col, ax=ax, hue='Номер_месяца', palette="coolwarm", legend=False, showfliers=False)
    
    ax.set_xlabel("Месяц")
    ylabel_text = "Выручка (руб)" if target_col == "Сумма" else "Количество (шт)"
    ax.set_ylabel(ylabel_text)
    ax.set_title("Анализ сезонности продаж")
    
    sns.despine()
    return fig

# --- 7. ABC Pie Chart ---
def draw_abc_pie_chart(abc_df):
    """Рисует круговую диаграмму, показывающую долю товаров в каждой ABC-категории."""
    if abc_df.empty:
        return None
    
    category_counts = abc_df['Категория'].value_counts()
    
    fig, ax = plt.subplots(figsize=(5, 4))
    fig.patch.set_alpha(0.0)
    
    colors = {'A': '#2E4B4F', 'B': '#4A3A2A', 'C': '#4F2E2E'}
    pie_colors = [colors.get(cat, '#888888') for cat in category_counts.index]
    
    wedges, texts, autotexts = ax.pie(
        category_counts, 
        labels=category_counts.index,
        autopct=lambda p: '{:.0f} поз.'.format(p * sum(category_counts) / 100),
        startangle=90, 
        colors=pie_colors,
        wedgeprops=dict(width=0.4, edgecolor='w'),
        textprops=dict(color="white", fontsize=10, weight="bold")
    )
    ax.set_title("Распределение позиций по группам", color="white", fontsize=12)
    return fig

# --- 8. Profitability ---
def analyze_profitability(df_filtered, min_sales=0):
    profit_col = 'Валовая прибыль (%)'
    if profit_col not in df_filtered.columns:
        return pd.DataFrame()

    # Исключаем записи с отсутствующей рентабельностью
    df_clean = df_filtered.dropna(subset=[profit_col])

    profitability = df_clean.groupby('Название').agg(
        Средняя_рентабельность=pd.NamedAgg(column=profit_col, aggfunc='mean'),
        Количество_продаж=pd.NamedAgg(column=profit_col, aggfunc='count')
    )
    # Фильтр по минимальному числу продаж, если требуется
    if min_sales > 0:
        profitability = profitability[profitability['Количество_продаж'] >= min_sales]
    
    profitability = profitability.sort_values(by='Средняя_рентабельность', ascending=False)
    return profitability.reset_index()