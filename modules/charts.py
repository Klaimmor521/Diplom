import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import matplotlib.ticker as ticker

# --- 1. Гистограмма по месяцам ---
def draw_revenue_bar(df_filtered, target_col):
    """Рисует гистограмму продаж по месяцам с русскими подписями."""
    monthly_sales = df_filtered.groupby('Месяц')[target_col].sum().reset_index()
    monthly_sales['Месяц_номер'] = pd.to_datetime(monthly_sales['Месяц']).dt.month

    all_months = pd.DataFrame({'Месяц_номер': range(1, 13)})
    monthly_sales = all_months.merge(monthly_sales, on='Месяц_номер', how='left')
    monthly_sales[target_col] = monthly_sales[target_col].fillna(0)  # отсутствующие = 0

    fig, ax = plt.subplots(figsize=(8, 6))
    barplot = sns.barplot(data=monthly_sales, x='Месяц_номер', y=target_col, ax=ax,
                          color=sns.color_palette("viridis", 1)[0], alpha=0.9)

    # Аннотации над столбцами
    for p in barplot.patches:
        height = p.get_height()
        if height > 0:
            ax.annotate(f"{height:.0f}",
                        (p.get_x() + p.get_width() / 2., height),
                        ha='center', va='center', xytext=(0, 9),
                        textcoords='offset points', fontsize=8, color="black", weight='medium')

    month_names_ru = ['Янв', 'Фев', 'Мар', 'Апр', 'Май', 'Июн',
                      'Июл', 'Авг', 'Сен', 'Окт', 'Ноя', 'Дек']
    ax.set_xticks(range(len(month_names_ru)))
    ax.set_xticklabels(month_names_ru, rotation=0)

    ax.set_xlabel("")
    ylabel_text = "Выручка (руб)" if target_col == "Сумма" else "Количество (шт)"
    ax.set_ylabel(ylabel_text, fontsize=12)
    return fig

# --- 2. Круговая диаграмма (Топ-10) ---
def draw_top_items_pie(df_filtered, target_col):
    """Рисует круговую диаграмму для Топ-10 в новом стиле."""
    top_items = df_filtered.groupby('Название')[target_col].sum().nlargest(10)
    fig, ax = plt.subplots(figsize=(10, 6))
    
    wedges, texts, autotexts = ax.pie(
        top_items, labels=None, autopct='%1.1f%%', 
        startangle=140, 
        colors=sns.color_palette("pastel", n_colors=len(top_items)),
        wedgeprops=dict(width=0.4, edgecolor='white', linewidth=2),
        textprops=dict(color="black", fontsize=9)
    )
    
    title_text = "Доля в выручке (₽)" if target_col == "Сумма" else "Доля в продажах (шт)"
    ax.set_title(title_text, pad=20)
    
    legend = ax.legend(wedges, top_items.index, title="Названия позиций", 
                       loc="center left", bbox_to_anchor=(1, 0, 0.5, 1),
                       frameon=False) # Убираем рамку у легенды
    plt.setp(legend.get_texts(), color="black")
    plt.setp(legend.get_title(), color="black", weight="bold")
    
    return fig

# --- 3. Сравнение год к году (YoY) ---
def draw_yoy_chart(df_filtered, target_col):
    """Рисует график сравнения год к году с русскими названиями месяцев."""
    df_yoy = df_filtered.copy()
    df_yoy['Год'] = df_yoy['Дата'].dt.year
    df_yoy['Номер_месяца'] = df_yoy['Дата'].dt.month

    df_yoy_grouped = df_yoy.groupby(['Год', 'Номер_месяца'])[target_col].sum().reset_index()

    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(data=df_yoy_grouped, x='Номер_месяца', y=target_col, hue='Год',
                palette="Set2", ax=ax, alpha=0.85)

    month_names_ru = ['Янв', 'Фев', 'Мар', 'Апр', 'Май', 'Июн',
                      'Июл', 'Авг', 'Сен', 'Окт', 'Ноя', 'Дек']
    ax.set_xticks(range(len(month_names_ru)))
    ax.set_xticklabels(month_names_ru, rotation=0)

    title_text = "Сравнение продаж по годам"
    ax.set_xlabel("Месяц")
    ylabel_text = "Выручка (руб)" if target_col == "Сумма" else "Количество (шт)"
    ax.set_ylabel(ylabel_text)
    ax.set_title(title_text, pad=15)

    if ax.get_legend() is not None:
        ax.get_legend().set_frame_on(False)

    return fig

# --- 4. График прогноза ---
def draw_forecast_chart(df_monthly, y, future_X, future_pred, target_type):
    """Рисует график прогноза с нумерацией месяцев, начинающейся с 1."""
    fig, ax = plt.subplots(figsize=(8, 5))

    palette = sns.color_palette("deep", 3)
    MA_COLOR, HISTORY_COLOR, FORECAST_COLOR = palette[2], palette[0], palette[1]

    x_history = df_monthly['Month_ID'] + 1
    x_future = future_X.ravel() + 1

    moving_average = y.rolling(window=3, center=True, min_periods=1).mean()
    sns.lineplot(x=x_history, y=moving_average, ax=ax, label="Скользящее среднее", color=MA_COLOR, linestyle=':', linewidth=2.5)
    
    sns.lineplot(x=x_history, y=y, ax=ax, label="Факт (История)", marker='o', color=HISTORY_COLOR, linewidth=2.5)
    
    ax.plot(x_future, future_pred, label="Прогноз (ML)", color=FORECAST_COLOR, linestyle='--', marker='s', linewidth=2)

    ax.fill_between(x_history, y, color=HISTORY_COLOR, alpha=0.1)

    last_y_hist = y.iloc[-1]
    ax.annotate(f'{last_y_hist:,.0f}', xy=(x_history.iloc[-1], last_y_hist), 
                xytext=(-15, 15), textcoords='offset points', weight='bold', color=HISTORY_COLOR)

    last_y_pred = future_pred[-1]
    last_x_pred = x_future[-1]
    ax.annotate(f'{last_y_pred:,.0f}', xy=(last_x_pred, last_y_pred), 
                xytext=(-15, 15), textcoords='offset points', weight='bold', color=FORECAST_COLOR)

    ax.set_title(f"Прогноз: {target_type}", pad=15)
    ax.set_xlabel("Номер месяца")
    ax.set_ylabel(target_type)
    total_months = int(x_future[-1] - x_history.iloc[0] + 1)
    
    if total_months > 24:
        step = max(1, total_months // 12)
    else:
        step = 1
    ax.xaxis.set_major_locator(ticker.MultipleLocator(step))
    ax.set_xlim(left=0.5)
    
    legend = ax.legend(loc="upper left", frameon=False)
    plt.setp(legend.get_texts(), color="black")
    
    return fig

# --- 5. Функция для ABC-анализа ---
def perform_abc_analysis(df_filtered):
    """Выполняет ABC-анализ номенклатуры по выручке."""
    if 'Сумма' not in df_filtered.columns or df_filtered['Сумма'].sum() == 0:
        return pd.DataFrame()

    item_revenue = df_filtered.groupby('Название')['Сумма'].sum().sort_values(ascending=False).reset_index()
    item_revenue.rename(columns={'Сумма': 'Выручка'}, inplace=True)
    
    total_revenue = item_revenue['Выручка'].sum()
    item_revenue['Доля в выручке'] = (item_revenue['Выручка'] / total_revenue)
    item_revenue['Накопительная доля'] = item_revenue['Доля в выручке'].cumsum()
    item_revenue['Доля в %'] = item_revenue['Доля в выручке'] * 100
    
    def assign_abc_category(share):
        if share <= 0.8: return 'A'
        if share <= 0.95: return 'B'
        return 'C'
    
    item_revenue['Категория'] = item_revenue['Накопительная доля'].apply(assign_abc_category)
    result_df = item_revenue[['Категория', 'Название', 'Выручка', 'Доля в %']]
    return result_df

# --- 6. Анализ сезонности (Box Plot) ---
def draw_seasonality_chart(df_filtered, target_col):
    """Рисует график анализа сезонности с подписями месяцев и средней линией."""
    df_season = df_filtered.copy()
    df_season['Номер_месяца'] = df_season['Дата'].dt.month

    fig, ax = plt.subplots(figsize=(8, 4))
    sns.boxplot(data=df_season, x='Номер_месяца', y=target_col, ax=ax, hue='Номер_месяца',
                palette="coolwarm", showfliers=False, legend=False)

    # Средняя линия
    overall_mean = df_season[target_col].mean()
    ax.axhline(y=overall_mean, color='grey', linestyle='--', linewidth=1, alpha=0.7, label=f'Среднее ({overall_mean:,.0f})')

    month_names_ru = ['Янв', 'Фев', 'Мар', 'Апр', 'Май', 'Июн',
                      'Июл', 'Авг', 'Сен', 'Окт', 'Ноя', 'Дек']
    ax.set_xticks(range(len(month_names_ru)))
    ax.set_xticklabels(month_names_ru, rotation=0)

    ax.set_xlabel("Месяц")
    ylabel_text = "Выручка (руб)" if target_col == "Сумма" else "Количество (шт)"
    ax.set_ylabel(ylabel_text)
    ax.set_title("Анализ сезонности продаж")
    ax.legend(loc='upper right')
    sns.despine()
    return fig

# --- 7. Круговая диаграмма для ABC-анализа ---
def draw_abc_pie_chart(abc_df):
    """Рисует круговую диаграмму для ABC-анализа в новом стиле."""
    if abc_df.empty: return None
    
    category_counts = abc_df['Категория'].value_counts()
    fig, ax = plt.subplots(figsize=(5, 4))
    
    colors = {'A': sns.color_palette("deep")[2], 'B': sns.color_palette("deep")[8], 'C': sns.color_palette("deep")[3]}
    pie_colors = [colors.get(cat, '#9CA3AF') for cat in category_counts.index]
    
    wedges, _, autotexts = ax.pie(
        category_counts, 
        labels=category_counts.index,
        autopct=lambda p: f'{p * sum(category_counts) / 100:.0f}\nпоз.',
        startangle=90, 
        colors=pie_colors,
        wedgeprops=dict(width=0.4, edgecolor='w', linewidth=3),
        pctdistance=0.8,
        textprops=dict(color="black", fontsize=10, weight="bold")
    )
    plt.setp(autotexts, size=10, weight="bold")
    ax.set_title("Распределение позиций по группам", fontsize=12)
    return fig

# --- 8. Функция для анализа рентабельности ---
def analyze_profitability(df_filtered, min_sales=0):
    """Анализирует рентабельность позиций."""
    profit_col = 'Валовая прибыль (%)'
    if profit_col not in df_filtered.columns:
        return pd.DataFrame()

    df_clean = df_filtered.dropna(subset=[profit_col])
    if df_clean.empty: return pd.DataFrame()

    profitability = df_clean.groupby('Название').agg(
        Средняя_рентабельность=pd.NamedAgg(column=profit_col, aggfunc='mean'),
        Количество_продаж=pd.NamedAgg(column=profit_col, aggfunc='count')
    )
    if min_sales > 0:
        profitability = profitability[profitability['Количество_продаж'] >= min_sales]
    
    return profitability.sort_values(by='Средняя_рентабельность', ascending=False).reset_index()