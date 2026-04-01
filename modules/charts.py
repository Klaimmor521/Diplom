import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid", palette="muted")

# --- 1. ГИСТОГРАММА ВЫРУЧКИ ПО МЕСЯЦАМ ---
def draw_revenue_bar(df_filtered):
    monthly_sales = df_filtered.groupby('Месяц')['Сумма'].sum().reset_index()
    fig, ax = plt.subplots(figsize=(8, 6))
    
    barplot = sns.barplot(data=monthly_sales, x='Месяц', y='Сумма', ax=ax, color="#5b9bd5", edgecolor="none", alpha=0.9)
    
    for p in barplot.patches:
        ax.annotate(format(p.get_height(), '.0f'), 
                    (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha = 'center', va = 'center', 
                    xytext = (0, 9), 
                    textcoords = 'offset points',
                    fontsize=8, color='#333333', rotation=45)
    
    ax.set_xlabel("")
    ax.set_ylabel("Выручка (руб)")
    plt.xticks(rotation=45)
    
    sns.despine(left=True)
    return fig

# --- 2. КРУГОВАЯ ДИАГРАММА (ТОП-10) ---
def draw_top_items_pie(df_filtered):
    top_items = df_filtered.groupby('Название')['Сумма'].sum().nlargest(10)
    fig, ax = plt.subplots(figsize=(8, 6))
    
    wedges, texts, autotexts = ax.pie(top_items, labels=top_items.index, autopct='%1.1f%%', 
                                      startangle=140, colors=sns.color_palette("pastel"), 
                                      wedgeprops=dict(width=0.4, edgecolor='w'))
    
    ax.set_title("Доля в выручке", pad=20)
    
    ax.legend(
        wedges, 
        top_items.index, 
        title="Названия позиций", 
        loc="center left", 
        bbox_to_anchor=(1, 0, 0.5, 1)
    )
    
    return fig

# --- 3. СРАВНЕНИЕ ГОД К ГОДУ (YoY) ---
def draw_yoy_chart(df_filtered):
    df_yoy = df_filtered.copy()
    df_yoy['Год'] = df_yoy['Дата'].dt.year
    df_yoy['Номер_месяца'] = df_yoy['Дата'].dt.month
    df_yoy_grouped = df_yoy.groupby(['Год', 'Номер_месяца'])['Сумма'].sum().reset_index()
    
    fig, ax = plt.subplots(figsize=(8, 4))
    
    sns.barplot(data=df_yoy_grouped, x='Номер_месяца', y='Сумма', hue='Год', palette='Set2', ax=ax, edgecolor="none", alpha=0.85)
    
    ax.set_xlabel("Месяц (1 - Январь, 12 - Декабрь)")
    ax.set_ylabel("Выручка (руб)")
    ax.set_title("Сравнение выручки по годам", pad=15)
    
    sns.despine(left=True)
    return fig

# --- 4. ПРОГНОЗ С ЗАЛИВКОЙ (Area Chart) ---
def draw_forecast_chart(df_monthly, y, future_X, future_pred, target_type):
    fig, ax = plt.subplots(figsize=(8, 5))
    
    ax.fill_between(df_monthly['Month_ID'], y, color="#4b8bbe", alpha=0.15)
    
    sns.lineplot(x=df_monthly['Month_ID'], y=y, ax=ax, label="Факт (История)", marker='o', color='#286090', linewidth=2.5)
    
    ax.plot(future_X, future_pred, label="Прогноз (ML)", color='#d62728', linestyle='--', marker='s', linewidth=2)
    
    last_x_hist = df_monthly['Month_ID'].iloc[-1]
    last_y_hist = y[-1]
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