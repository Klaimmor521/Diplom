import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
import sys

sys.stdout.reconfigure(encoding='utf-8')

def generate_livesklad_exact_copy():
    print("Генерируем данные (LiveSklad)...")
    
    # Генерация
    rows = 3500  # Строк
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2025, 5, 1)

    catalog = [
        # Услуги
        ("Диагностика", 500, 0, "Заказ"),
        ("Чистка системы охлаждения с заменой термопасты", 1500, 100, "Заказ"),
        ("Восстановление цепи питания", 4500, 500, "Заказ"),
        ("Установка Windows 10/11 + Драйвера", 1200, 0, "Заказ"),
        ("Замена разъема зарядки (MicroUSB/Type-C)", 1200, 100, "Заказ"),
        ("Прошивка Android/iOS", 1000, 0, "Заказ"),
        ("Наклейка защитного стекла (Услуга)", 200, 0, "Заказ"),
        
        # Товары запчасти
        ("15.6\" WXGA HD LED Глянцевый 40pin (NT156WH)", 4500, 3200, "Заказ"), # Матрица
        ("Дисплей iPhone 11 (Копия AAA)", 2500, 1400, "Заказ"),
        ("Дисплей iPhone 11 (Оригинал переклей)", 5500, 3800, "Заказ"),
        ("Аккумулятор Samsung Galaxy A51 (EB-BA515)", 1200, 550, "Заказ"),
        ("Жесткий диск Seagate BarraCuda 1TB", 3800, 2900, "Продажа"),
        ("SSD накопитель Kingston 240GB", 2200, 1500, "Продажа"),
        
        # Товары аксессуары
        ("Защитное стекло 3D для iPhone 11/XR", 500, 50, "Продажа"),
        ("Чехол силиконовый прозрачный", 300, 60, "Продажа"),
        ("Кабель USB - Lightning (Foxconn)", 600, 150, "Продажа"),
        ("Блок питания Apple 20W (Copy)", 1200, 400, "Продажа"),
        ("16GB USB 2.0 Flash Drive SmartBuy", 450, 200, "Продажа")
    ]

    employees = ["Петров Роман", "Иванов Сергей", "Сидоров Алексей", "Администратор"]

    data = []

    for _ in range(rows):
        # Генерация даты
        delta_days = (end_date - start_date).days
        random_day = start_date + timedelta(days=random.randrange(delta_days))
        random_time = timedelta(hours=random.randint(10, 19), minutes=random.randint(0, 59))
        final_date = random_day + random_time
        
        # Выбор товара/услуги (С учетом популярности)
        item = random.choices(catalog, weights=[
            10, 8, 5, 8, 8, 5, 10,  # Услуги
            3, 5, 3, 5, 2, 3,       # Запчасти
            15, 15, 10, 5, 8        # Аксессуары
        ], k=1)[0]
        
        name = item[0]
        base_price = item[1]
        base_cost = item[2]
        doc_type_base = item[3] # "Продажа" или "Заказ"

        # Вариация цены
        price_variation = random.uniform(0.9, 1.1)
        price = int(base_price * price_variation)
        
        # Вариация закупки
        cost_variation = random.uniform(0.95, 1.05)
        cost = int(base_cost * cost_variation)

        # Количество
        qty = random.choices([1, 2, 3], weights=[90, 8, 2])[0]

        # Расчеты
        total_sum = price * qty # Сумма
        total_cost = cost * qty # Общая себестоимость
        gross_profit = total_sum - total_cost # Валовая прибыль

        # Тип документа
        if doc_type_base == "Заказ":
            doc_type = "Заказ"
        else:
            doc_type = random.choice(["Продажа", "Заказ"])

        # Формат даты: "03.09.2024 - 12:42"
        date_str = final_date.strftime("%d.%m.%Y - %H:%M")
        
        employee = random.choice(employees)

        data.append([
            date_str,      # Дата
            doc_type,      # Тип документа
            name,          # Название
            qty,           # Кол-во
            price,         # Цена
            total_sum,     # Сумма
            gross_profit,  # Валовая прибыль
            employee       # Сотрудник
        ])

    # DataFrame
    columns = [
        "Дата", 
        "Тип документа", 
        "Название", 
        "Количество", 
        "Цена", 
        "Сумма", 
        "Валовая прибыль (руб)", 
        "Сотрудник"
    ]
    
    df = pd.DataFrame(data, columns=columns)

    # Сортировка по дате
    df['_temp_sort'] = pd.to_datetime(df['Дата'], format="%d.%m.%Y - %H:%M")
    df = df.sort_values('_temp_sort')
    df = df.drop(columns=['_temp_sort'])

    # Сохраняем в CSV
    filename = "data/livesklad_export.csv"
    df.to_csv(filename, index=False)
    print(f"Файл '{filename}' успешно создан! Структура совпадает со скриншотом.")

if __name__ == "__main__":
    generate_livesklad_exact_copy()