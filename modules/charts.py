import matplotlib.pyplot as plt
import seaborn as sns

def draw_revenue_bar(df_filtered):
    monthly_sales = df_filtered.groupby('Месяц')['Сумма'].sum().reset_index()
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=monthly_sales, x='Месяц', y='Сумма', ax=ax, color="#4b8bbe")
    ax.set_xlabel("Месяц")
    ax.set_ylabel("Выручка (руб)")
    plt.xticks(rotation=45)
    return fig

def draw_top_items_pie(df_filtered):
    top_items = df_filtered.groupby('Название')['Сумма'].sum().nlargest(10)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.pie(top_items, labels=top_items.index, autopct='%1.1f%%', startangle=140, colors=sns.color_palette("pastel"))
    ax.set_title("Доля в выручке")
    return fig

def draw_yoy_chart(df_filtered):
    df_yoy = df_filtered.copy()
    df_yoy['Год'] = df_yoy['Дата'].dt.year
    df_yoy['Номер_месяца'] = df_yoy['Дата'].dt.month
    df_yoy_grouped = df_yoy.groupby(['Год', 'Номер_месяца'])['Сумма'].sum().reset_index()
    
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.barplot(data=df_yoy_grouped, x='Номер_месяца', y='Сумма', hue='Год', palette='viridis', ax=ax)
    ax.set_xlabel("Месяц (1 - Январь, 12 - Декабрь)")
    ax.set_ylabel("Выручка (руб)")
    ax.set_title("Сравнение выручки по годам")
    return fig

def draw_forecast_chart(df_monthly, y, future_X, future_pred, target_type):
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.lineplot(x=df_monthly['Month_ID'], y=y, ax=ax, label="Факт (История)", marker='o', color='black')
    ax.plot(future_X, future_pred, label="Прогноз (ML)", color='red', linestyle='--', marker='x')
    ax.set_title(f"Прогноз: {target_type}")
    ax.set_xlabel("Номер месяца (с начала наблюдений)")
    ax.set_ylabel(target_type)
    ax.legend()
    ax.grid(True)
    return fig