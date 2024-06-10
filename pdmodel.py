import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest, RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Função para limpar os dados
def clean_data(df):
    filtered_df = df[(df['PARAMETER'] == 'Z1MaxCurrent') & (df['EQUIPMENT'] == 'WLPBSG-0002')]
    cleaned_df = filtered_df[['EQUIPMENT', 'DATETIME', 'VALUE']]
    return cleaned_df

# Caminho para o arquivo Excel
file_path = r'C:\Users\luis_\OneDrive\Ambiente de Trabalho\dados_amkor.xlsx'


# Leitura dos dados do Excel
excel_data = pd.ExcelFile(file_path)
cleaned_dataframes = []

# Carregar e limpar os dados de cada planilha
for sheet_name in excel_data.sheet_names:
    df = pd.read_excel(file_path, sheet_name=sheet_name)
    cleaned_df = clean_data(df)
    cleaned_dataframes.append(cleaned_df)

# Concatenar todos os dados limpos em um único DataFrame
df = pd.concat(cleaned_dataframes, ignore_index=True)

# Remover espaços em branco nas strings da coluna DATETIME
df['DATETIME'] = df['DATETIME'].str.strip()
df['DATETIME'] = pd.to_datetime(df['DATETIME'], errors='coerce')

# Converter a coluna 'VALUE' para números
df['VALUE'] = pd.to_numeric(df['VALUE'], errors='coerce')

# Remover valores NaN criados durante a conversão
df = df.dropna(subset=['DATETIME', 'VALUE'])

# Ordenar os dados por DATETIME
df = df.sort_values(by='DATETIME')

# Adicionar uma coluna de índice temporal
df['TimeIndex'] = (df['DATETIME'] - df['DATETIME'].min()).dt.total_seconds() / 60
df['TimeIndex'] = df['TimeIndex'].astype(int)

# Feature Engineering
df['Hour'] = df['DATETIME'].dt.hour
df['DayOfWeek'] = df['DATETIME'].dt.dayofweek
df['Month'] = df['DATETIME'].dt.month

# Preparação dos dados para detecção de anomalias
features = ['TimeIndex', 'Hour', 'DayOfWeek', 'Month', 'VALUE']
X = df[features]

# Padronização das features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Modelo de detecção de anomalias
model = IsolationForest(n_estimators=100, contamination=0.01, random_state=42)
anomaly_labels = model.fit_predict(X_scaled)

# Marcar anomalias (onde anomaly == -1)
df['anomaly'] = np.where(anomaly_labels == -1, 1, 0)

# Dividir os dados em conjuntos de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X.drop('VALUE', axis=1), df['VALUE'], test_size=0.2, random_state=42)

# Função para calcular e exibir métricas
def evaluate_model(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    return mse, mae, rmse, r2

# Estudo de modelos preditivos

# Modelo de Regressão Linear
lr_model = LinearRegression()
mse_lr, mae_lr, rmse_lr, r2_lr = evaluate_model(lr_model, X_train, X_test, y_train, y_test)
print(f'Regressão Linear - MSE: {mse_lr:.2f}, MAE: {mae_lr:.2f}, RMSE: {rmse_lr:.2f}, R²: {r2_lr:.2f}')

# Modelo de Random Forest
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
mse_rf, mae_rf, rmse_rf, r2_rf = evaluate_model(rf_model, X_train, X_test, y_train, y_test)
print(f'Random Forest - MSE: {mse_rf:.2f}, MAE: {mae_rf:.2f}, RMSE: {rmse_rf:.2f}, R²: {r2_rf:.2f}')

# Modelo de Gradient Boosting
gb_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
mse_gb, mae_gb, rmse_gb, r2_gb = evaluate_model(gb_model, X_train, X_test, y_train, y_test)
print(f'Gradient Boosting - MSE: {mse_gb:.2f}, MAE: {mae_gb:.2f}, RMSE: {rmse_gb:.2f}, R²: {r2_gb:.2f}')

# Modelo de Support Vector Regressor (SVR)
svr_model = SVR()
mse_svr, mae_svr, rmse_svr, r2_svr = evaluate_model(svr_model, X_train, X_test, y_train, y_test)
print(f'SVR - MSE: {mse_svr:.2f}, MAE: {mae_svr:.2f}, RMSE: {rmse_svr:.2f}, R²: {r2_svr:.2f}')

# Modelo de XGBoost
xgb_model = XGBRegressor(n_estimators=100, random_state=42)
mse_xgb, mae_xgb, rmse_xgb, r2_xgb = evaluate_model(xgb_model, X_train, X_test, y_train, y_test)
print(f'XGBoost - MSE: {mse_xgb:.2f}, MAE: {mae_xgb:.2f}, RMSE: {rmse_xgb:.2f}, R²: {r2_xgb:.2f}')


# Seleção do algoritmo mais adequado
model_metrics = {
    'Linear Regression': {'MSE': mse_lr, 'MAE': mae_lr, 'RMSE': rmse_lr, 'R²': r2_lr},
    'Random Forest': {'MSE': mse_rf, 'MAE': mae_rf, 'RMSE': rmse_rf, 'R²': r2_rf},
    'Gradient Boosting': {'MSE': mse_gb, 'MAE': mae_gb, 'RMSE': rmse_gb, 'R²': r2_gb},
    'SVR': {'MSE': mse_svr, 'MAE': mae_svr, 'RMSE': rmse_svr, 'R²': r2_svr},
    'XGBoost': {'MSE': mse_xgb, 'MAE': mae_xgb, 'RMSE': rmse_xgb, 'R²': r2_xgb},
}

best_model_name = min(model_metrics, key=lambda x: model_metrics[x]['MSE'])
print(f'O modelo mais adequado é: {best_model_name}')

# Implementação do algoritmo selecionado (exemplo com XGBoost)
best_model = xgb_model
best_model.fit(X_train, y_train)

# Validação do modelo
y_pred = best_model.predict(X_test)

# Plotar os gráficos


# Gráfico 1: Anomalias detectadas
plt.figure(figsize=(10, 6))
plt.plot(df['DATETIME'], df['VALUE'], color='blue', label='Normal')
plt.scatter(df[df['anomaly'] == 1]['DATETIME'], df[df['anomaly'] == 1]['VALUE'], color='red', label='Anomaly')
plt.xlabel('Date')
plt.ylabel('MaxCurrent')
plt.title('Anomaly Detection in Max Current')
plt.legend()
plt.grid(True)
plt.show()

# Gráfico 2: Comparação dos valores reais e previstos (tamanho total)
plt.figure(figsize=(10, 6))
plt.plot(y_test.index, y_test, label='True Values', color='blue')
plt.plot(y_test.index, y_pred, label='Predicted Values', color='red')
plt.xlabel('Index')
plt.ylabel('MaxCurrent')
plt.title('Comparison of True and Predicted Values (Full Data)')
plt.legend()
plt.grid(True)
plt.show()

# Gráfico 3: Comparação dos valores reais e previstos (subconjunto)
subset_size = 200  # tamanho do subconjunto para visualização
plt.figure(figsize=(10, 6))
plt.plot(y_test.head(subset_size).index, y_test.head(subset_size), label='True Values', color='blue')
plt.plot(y_test.head(subset_size).index, y_pred[:subset_size], label='Predicted Values', color='orange')
plt.xlabel('Index')
plt.ylabel('MaxCurrent')
plt.title('Comparison of True and Predicted Values (Subset)')
plt.legend()
plt.grid(True)
plt.show()
