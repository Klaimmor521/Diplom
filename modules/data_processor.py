import pandas as pd
import streamlit as st

@st.cache_data
def load_data(file_path_or_buffer):
    try:
        # 1. Проверяем, это загруженный файл (с сайта) или наш тестовый (с компьютера)
        if hasattr(file_path_or_buffer, 'name'):
            file_name = file_path_or_buffer.name
        else:
            file_name = str(file_path_or_buffer)

        # 2. Читаем файл в зависимости от его расширения
        if file_name.endswith('.xlsx') or file_name.endswith('.xls'):
            df = pd.read_excel(file_path_or_buffer)
        elif file_name.endswith('.csv'):
            df = pd.read_csv(file_path_or_buffer)
        else:
            st.error("❌ Неподдерживаемый формат файла. Загрузите CSV или Excel.")
            return None

        if 'Тип документа' in df.columns:
            df = df[df['Тип документа'].isin(['Продажа', 'Установка в заказ', 'Заказ'])].copy()

        # Очистка Даты (убираем тире, если оно есть в LiveSklad)
        df['Дата'] = df['Дата'].astype(str)
        df['Дата'] = df['Дата'].str.replace('-', '').str.strip()
        df['Дата'] = pd.to_datetime(df['Дата'], dayfirst=True)
            
        df['Месяц'] = df['Дата'].dt.to_period('M').astype(str)

        if df['Валовая прибыль (руб)'].dtype == object: # Если прочиталось как текст
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