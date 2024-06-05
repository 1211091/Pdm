import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Função para limpar os dados
def clean_data(df):
    """
    Função para filtrar e limpar os dados.
    Mantém apenas as colunas 'EQUIPMENT', 'DATETIME', e 'VALUE'
    onde 'PARAMETER' é 'Z1MaxCurrent' e 'EQUIPMENT' é 'WLPBSG-0002'.
    """
    filtered_df = df[(df['PARAMETER'] == 'Z1MaxCurrent') & (df['EQUIPMENT'] == 'WLPBSG-0002')]
    cleaned_df = filtered_df[['EQUIPMENT', 'DATETIME', 'VALUE']]
    return cleaned_df

# Caminho para o arquivo Excel
file_path = r'C:\Users\luis_\OneDrive\Ambiente de Trabalho\dados_amkor.xlsx'

# Leitura dos dados do Excel
excel_data = pd.ExcelFile(file_path)

# Lista para armazenar os DataFrames limpos
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

# Conversão da coluna de Timestamp para datetime
df['DATETIME'] = pd.to_datetime(df['DATETIME'], format='mixed')

# Adicionando uma coluna de índice temporal
df['TimeIndex'] = (df['DATETIME'] - df['DATETIME'].min()).dt.total_seconds() / 60
df['TimeIndex'] = df['TimeIndex'].astype(int)

# Adicionando features de tempo
df['Hour'] = df['DATETIME'].dt.hour
df['Day'] = df['DATETIME'].dt.day
df['Month'] = df['DATETIME'].dt.month

# Preparação dos dados para o modelo
X = df[['TimeIndex', 'Hour', 'Day', 'Month']]
y = df['VALUE']

# Normalização dos dados
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Divisão dos dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Definição do pipeline para o modelo de Decision Tree
dt_pipeline = Pipeline([
    ('dt', DecisionTreeRegressor(random_state=42))
])

# Definição do pipeline para o modelo de Random Forest
rf_pipeline = Pipeline([
    ('rf', RandomForestRegressor(n_estimators=100, random_state=42))
])

# Definição dos parâmetros para busca de hiperparâmetros
dt_params = {
    'dt__max_depth': [None, 5, 10],
    'dt__min_samples_split': [2, 5, 10],
    'dt__min_samples_leaf': [1, 5, 10]
}

rf_params = {
    'rf__n_estimators': [50, 100, 200],
    'rf__max_depth': [None, 5, 10],
    'rf__min_samples_split': [2, 5, 10],
    'rf__min_samples_leaf': [1, 5, 10]
}

# Busca de hiperparâmetros para o modelo de Decision Tree
dt_grid = GridSearchCV(dt_pipeline, dt_params, cv=5, scoring='neg_mean_squared_error')
dt_grid.fit(X_train, y_train)

# Treinamento do modelo de Decision Tree com os melhores hiperparâmetros
dt_model = dt_grid.best_estimator_
dt_model.fit(X_train, y_train)

# Previsões
y_pred_dt = dt_model.predict(X_test)

# Avaliação do modelo de Decision Tree
mse_dt = mean_squared_error(y_test, y_pred_dt)
r2_dt = r2_score(y_test, y_pred_dt)
mae_dt = mean_absolute_error(y_test, y_pred_dt)

print(f'Decision Tree MSE: {mse_dt}')
print(f'Decision Tree R2: {r2_dt}')
print(f'Decision Tree MAE: {mae_dt}')

# Busca de hiperparâmetros para o modelo de Random Forest
rf_grid = GridSearchCV(rf_pipeline, rf_params, cv=5, scoring='neg_mean_squared_error')
rf_grid.fit(X_train, y_train)

# Treinamento do modelo de Random Forest com os melhores hiperparâmetros
rf_model = rf_grid.best_estimator_
rf_model.fit(X_train, y_train)

# Previsões
y_pred_rf = rf_model.predict(X_test)

# Avaliação do modelo de Random Forest
mse_rf = mean_squared_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)
mae_rf = mean_absolute_error(y_test, y_pred_rf)

print(f'Random Forest MSE: {mse_rf}')
print(f'Random Forest R2: {r2_rf}')
print(f'Random Forest MAE: {mae_rf}')

# Filtragem dos dados para os meses de abril e maio
april_may_data = df[(df['Month'] == 4) | (df['Month'] == 5)]

# Visualização do MaxCurrent para os meses de abril e maio
plt.figure(figsize=(12, 6))
plt.plot(april_may_data['DATETIME'], april_may_data['VALUE'], label='Real')
plt.legend()
plt.title('MaxCurrent para abril e maio')
plt.xlabel('Data')
plt.ylabel('MaxCurrent')
plt.grid()
plt.show()

# Visualização das previsões do modelo de Decision Tree
plt.figure(figsize=(12, 6))
plt.plot(y_test, label='Real')
plt.plot(y_pred_dt, label='Decision Tree')
plt.legend()
plt.title('Previsões do modelo de Decision Tree')
plt.xlabel('Índice')
plt.ylabel('MaxCurrent')
plt.grid()
plt.show()

# Visualização das previsões do modelo de Random Forest
plt.figure(figsize=(12, 6))
plt.plot(y_test, label='Real')
plt.plot(y_pred_rf, label='Random Forest')
plt.legend()
plt.title('Previsões do modelo de Random Forest')
plt.xlabel('Índice')
plt.ylabel('MaxCurrent')
plt.grid()
plt.show()