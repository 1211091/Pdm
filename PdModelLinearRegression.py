import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.seasonal import seasonal_decompose

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
df['DATETIME'] = pd.to_datetime(df['DATETIME'], format='mixed')

# Agregar os dados por dia, calculando a média de VALUE
df_daily = df.resample('D', on='DATETIME').mean(numeric_only=True).reset_index()
df_daily.columns = ['DATETIME', 'VALUE']  # rename the columns

# Removendo linhas com valores nulos em 'VALUE'
df_daily.dropna(subset=['VALUE'], inplace=True)

# Análise de tendências e sazonalidade
result = seasonal_decompose(df_daily.set_index('DATETIME')['VALUE'], model='additive', period=30)
result.plot()
plt.show()

# Adicionando uma coluna de índice temporal
df_daily['TimeIndex'] = (df_daily['DATETIME'] - df_daily['DATETIME'].min()).dt.total_seconds() / 60
df_daily['TimeIndex'] = df_daily['TimeIndex'].astype(int)

# Feature Engineering
df_daily['Hour'] = df_daily['DATETIME'].dt.hour
df_daily['DayOfWeek'] = df_daily['DATETIME'].dt.dayofweek
df_daily['Month'] = df_daily['DATETIME'].dt.month

# Preparação dos dados para detecção de anomalias
features = ['TimeIndex', 'Hour', 'DayOfWeek', 'Month', 'VALUE']
X = df_daily[features]

# Padronização das features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Modelo de detecção de anomalias
model = IsolationForest(n_estimators=100, contamination=0.01, random_state=42)
df_daily['anomaly'] = model.fit_predict(X_scaled)

# Marcar anomalias (onde anomaly == -1)
df_daily['anomaly'] = df_daily['anomaly'].map({1: 0, -1: 1})

# Visualização dos dados e anomalias
plt.figure(figsize=(15, 8))
plt.plot(df_daily['DATETIME'], df_daily['VALUE'], label='MaxCurrent', color='blue')
plt.scatter(df_daily[df_daily['anomaly'] == 1]['DATETIME'], df_daily[df_daily['anomaly'] == 1]['VALUE'], color='red', label='Anomaly', s=50)
plt.xlabel('Datetime')
plt.ylabel('MaxCurrent')
plt.title('Anomaly Detection in MaxCurrent (Daily)')
plt.legend()
plt.grid(True)
plt.show()

# Visualização das anomalias em um gráfico de dispersão
plt.figure(figsize=(15, 8))
plt.scatter(df_daily['TimeIndex'], df_daily['VALUE'], c=df_daily['anomaly'], cmap='coolwarm', label='Anomaly')
plt.xlabel('TimeIndex')
plt.ylabel('MaxCurrent')
plt.title('Anomaly Detection in MaxCurrent (Scatter Plot)')
plt.colorbar(label='Anomaly')
plt.grid(True)
plt.xlim(df_daily['TimeIndex'].min(), df_daily['TimeIndex'].min() + 60 * 24 * 30)  # Ajustando o limite do eixo x para 30 dias
plt.show()