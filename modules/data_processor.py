import pandas as pd
import streamlit as st

@st.cache_data
def load_data(file_path_or_buffer):
    try:
        df = pd.read_csv(file_path_or_buffer)
        # Обработка даты (пробуем формат с тире, если нет - обычный)
        try:
            df['Дата'] = pd.to_datetime(df['Дата'], format="%d.%m.%Y - %H:%M")
        except:
            df['Дата'] = pd.to_datetime(df['Дата'], dayfirst=True)
            
        df['Месяц'] = df['Дата'].dt.to_period('M').astype(str)
        return df
    except Exception as e:
        return None

def classify_smart(row, use_smart_sort):
    if not use_smart_sort:
        if row['Тип документа'] == 'Продажа': return 'Товар'
        return 'Услуга'
        
    name = str(row['Название']).lower()
    services =['диагностика', 'пайк', 'чистк', 'установк', 'замен', 'восстановлени', 'прошивк', 'наклейк']
    parts =['дисплей', 'акб', 'аккумулятор', 'разъем', 'корпус', 'ssd', 'диск', 'матрица', 'led']
    accessories =['чехол', 'стекло', 'пленк', 'кабель', 'usb', 'блок', 'adapter', 'drive']
    
    if any(word in name for word in services): return 'Услуги'
    if any(word in name for word in parts): return 'Запчасти'
    if any(word in name for word in accessories): return 'Аксессуары'
    return 'Прочее'