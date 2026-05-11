import pandas as pd
import streamlit as st

@st.cache_data
def load_data(file_path_or_buffer):
    try:
        if hasattr(file_path_or_buffer, 'name'):
            file_name = file_path_or_buffer.name
        else:
            file_name = str(file_path_or_buffer)

        if file_name.endswith('.xlsx') or file_name.endswith('.xls'):
            df = pd.read_excel(file_path_or_buffer)
        elif file_name.endswith('.csv'):
            df = pd.read_csv(file_path_or_buffer)
        else:
            st.error("❌ Неподдерживаемый формат файла. Загрузите CSV или Excel.")
            return None

        df = df[df['Тип документа'].isin(['Продажа', 'Установка в заказ', 'Заказ'])].copy()

        if 'Статус' in df.columns:
            # Заполняем пустые статусы, чтобы они не мешали фильтрации
            df['Статус'] = df['Статус'].fillna('')
            df = df[ (df['Тип документа'] == 'Продажа') | (df['Статус'] == 'Выдан') ]

        # Очистка Даты
        df['Дата'] = df['Дата'].astype(str).str.split(' - ').str[0].str.strip()
        df['Дата'] = pd.to_datetime(df['Дата'], format='%d.%m.%Y', errors='coerce')
        # Удаляем строки, где дату не удалось распарсить
        df = df.dropna(subset=['Дата'])
            
        df['Месяц'] = df['Дата'].dt.to_period('M').astype(str)

        if df['Валовая прибыль (руб)'].dtype == object:
            df['Валовая прибыль (руб)'] = df['Валовая прибыль (руб)'].astype(str).str.replace(',', '.').astype(float)
            
        if df['Сумма'].dtype == object:
            df['Сумма'] = df['Сумма'].astype(str).str.replace(',', '.').astype(float)
        
        return df
        
    except Exception as e:
        st.error(f"Ошибка при чтении файла: {e}")
        return None

def classify_smart(row, use_smart_sort):
    if not use_smart_sort:
        if row['Тип документа'] == 'Продажа': 
            return 'Товар'
        return 'Услуга'
        
    name = str(row['Название']).lower()
    services =['диагностика', 'пайк', 'чистк', 'установк', 'замен', 'восстановлени', 'прошивк', 'наклейк']
    parts =['дисплей', 'акб', 'аккумулятор', 'разъем', 'корпус', 'ssd', 'диск', 'матрица', 'led']
    accessories =['чехол', 'стекло', 'пленк', 'кабель', 'usb', 'блок', 'adapter', 'drive']
    
    if any(word in name for word in services): return 'Услуги'
    if any(word in name for word in parts): return 'Запчасти'
    if any(word in name for word in accessories): return 'Аксессуары'
    return 'Прочее'