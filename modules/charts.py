import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid", palette="muted")

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
    # Считаем выручку по каждому товару
    item_revenue = df_filtered.groupby('Название')['Сумма'].sum().sort_values(ascending=False).reset_index()
    
    # Считаем долю каждого товара и накопительную долю
    item_revenue['Доля'] = item_revenue['Сумма'] / item_revenue['Сумма'].sum()
    item_revenue['Накопительная доля'] = item_revenue['Доля'].cumsum()
    
    # Присваиваем категории A, B, C
    def assign_abc_category(share):
        if share <= 0.8:
            return 'A (Ключевые)'
        elif share <= 0.95:
            return 'B (Стабильные)'
        else:
            return 'C (Незначительные)'
    
    item_revenue['ABC Категория'] = item_revenue['Накопительная доля'].apply(assign_abc_category)
    return item_revenue

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