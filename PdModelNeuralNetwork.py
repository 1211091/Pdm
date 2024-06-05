import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor

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

# Feature engineering - adicionando mais características temporais
df['Year'] = df['DATETIME'].dt.year
df['Month'] = df['DATETIME'].dt.month
df['Day'] = df['DATETIME'].dt.day
df['DayOfWeek'] = df['DATETIME'].dt.dayofweek
df['Hour'] = df['DATETIME'].dt.hour
df['Minute'] = df['DATETIME'].dt.minute
df['TimeIndex'] = (df['DATETIME'] - df['DATETIME'].min()).dt.total_seconds() / 60
df['TimeIndex'] = df['TimeIndex'].astype(int)

# Preparação dos dados para o modelo
X = df[['TimeIndex', 'Year', 'Month', 'Day', 'DayOfWeek', 'Hour', 'Minute']]
y = df['VALUE']

# Normalização dos dados
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Divisão dos dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Definição do modelo de rede neural
model = MLPRegressor(max_iter=2500, random_state=42)

# Pipeline para combinar normalização e modelo
pipeline = Pipeline([
    ('scaler', scaler),
    ('model', model)
])

# GridSearchCV para encontrar os melhores hiperparâmetros
param_grid = {
    'model__alpha': [0.0001, 0.001, 0.01, 0.1],
    'model__learning_rate': ['constant', 'invscaling', 'adaptive'],
    'model__hidden_layer_sizes': [(50,), (100,), (200,)],
    'model__activation': ['relu', 'tanh', 'logistic'],
    'model__solver': ['adam', 'lbfgs', 'sgd']
}

grid_search = GridSearchCV(pipeline, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error', verbose=2, n_jobs=-1)

# Treinamento do modelo com GridSearchCV
grid_search.fit(X_train, y_train)

# Melhores parâmetros encontrados
print("Melhores parâmetros encontrados:")
print(grid_search.best_params_)

# Previsões
y_pred = grid_search.predict(X_test)

# Avaliação do modelo
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print(f'MSE: {mse}')
print(f'R2: {r2}')
print(f'MAE: {mae}')

# Visualização das previsões versus valores reais
plt.figure(figsize=(10, 6))
plt.scatter(X_test[:, 0], y_test, color='blue', label='Real', alpha=0.5)
plt.scatter(X_test[:, 0], y_pred, color='red', label='Predicted', alpha=0.5)
plt.xlabel('TimeIndex')
plt.ylabel('MaxCurrent')
plt.title('Real vs Predicted')
plt.legend()
plt.grid(True)
plt.show()
