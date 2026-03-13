import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

def run_prediction(df_monthly, target_col, future_months):
    df_monthly['Month_ID'] = range(len(df_monthly))
    X = df_monthly[['Month_ID']].values 
    y = df_monthly[target_col].values

    model = LinearRegression()
    model.fit(X, y)

    # Метрики
    y_pred_train = model.predict(X)
    r2 = max(0, r2_score(y, y_pred_train)) * 100
    mae = mean_absolute_error(y, y_pred_train)

    # Прогноз
    last_id = df_monthly['Month_ID'].max() 
    future_X = np.arange(last_id + 1, last_id + 1 + future_months).reshape(-1, 1)
    future_pred = model.predict(future_X)

    return X, y, future_X, future_pred, r2, mae