import os
import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.preprocessing import LabelEncoder

# Carregar automaticamente o dataset
DATA_PATH = "food_coded.csv"  # Caminho fixo do arquivo

def carregar_dados():
    try:
        df = pd.read_csv(DATA_PATH)
        return df
    except FileNotFoundError:
        st.error("Erro: O arquivo 'food_coded.csv' não foi encontrado no diretório.")
        return None

# Título da aplicação
st.title("Análise de Dados Categóricos")

df = carregar_dados()
if df is not None:
    # Selecionando colunas categóricas
    object_columns = df.select_dtypes(include=['object']).columns

    # Exibir as colunas categóricas
    st.subheader("Colunas Categóricas")
    st.write(df[object_columns].head())

    # Escolher uma coluna para visualizar os valores únicos
    coluna_selecionada = st.selectbox("Selecione uma coluna categórica", object_columns)

    if coluna_selecionada:
        # Exibir contagem dos valores únicos
        st.subheader(f"Distribuição de valores - {coluna_selecionada}")
        st.write(df[coluna_selecionada].value_counts())

        # Criar gráfico de barras
        st.bar_chart(df[coluna_selecionada].value_counts())


# Configuração do título
st.title("📊 Análise de Dados de Diabetes")

# Caminho do arquivo CSV
diabetes_csv = "diabetes_prediction_dataset.csv"

# Verifica se o arquivo existe ou pede upload
if os.path.exists(diabetes_csv):
    data = pd.read_csv(diabetes_csv)
    st.success("✅ Arquivo carregado automaticamente.")
else:
    file = st.file_uploader("Carregue o arquivo 'diabetes_prediction_dataset.csv'", type="csv")
    if file:
        data = pd.read_csv(file)
        st.success("✅ Arquivo carregado com sucesso!")
    else:
        st.warning("⚠️ Nenhum arquivo carregado. Aguarde o upload.")
        st.stop()  # Interrompe o código se não houver arquivo

# Exibe informações básicas
st.subheader("🔍 Primeiras Linhas do Dataset")
st.dataframe(data.head())

st.subheader("📈 Estatísticas Descritivas")
st.write(data.describe())

st.subheader("ℹ️ Informações do Dataset")
st.text(str(data.info()))  # Converte info() para string e exibe no Streamlit

# 📊 **Gráficos**
st.subheader("📊 Contagem por Diabetes")
fig, ax = plt.subplots()
sns.countplot(data=data, x='diabetes', ax=ax)
st.pyplot(fig)

st.subheader("📊 Contagem por Gênero")
fig, ax = plt.subplots()
sns.countplot(data=data, x='gender', ax=ax)
st.pyplot(fig)

# Histogramas e Boxplots de Idade
st.subheader("📊 Distribuição da Idade")
fig, axes = plt.subplots(1, 2, figsize=(15, 5))
sns.histplot(data['age'], bins=9, ax=axes[0])
sns.boxplot(y=data['age'], ax=axes[1])
st.pyplot(fig)

# Contagem de Hipertensão
st.subheader("📊 Contagem de Pacientes com Hipertensão")
fig, ax = plt.subplots()
sns.countplot(data=data, x='hypertension', ax=ax)
st.pyplot(fig)

# Contagem de Problemas Cardíacos
st.subheader("📊 Contagem de Pacientes com Problemas Cardíacos")
fig, ax = plt.subplots()
sns.countplot(data=data, x='heart_disease', ax=ax)
st.pyplot(fig)

# Contagem por Histórico de Tabagismo
st.subheader("📊 Contagem por Histórico de Tabagismo")
fig, ax = plt.subplots()
sns.countplot(data=data, x='smoking_history', order=data['smoking_history'].value_counts().index, ax=ax)
st.pyplot(fig)

# Histogramas e Boxplots para HbA1c Level
st.subheader("📊 Nível de HbA1c")
fig, axes = plt.subplots(1, 2, figsize=(15, 5))
sns.histplot(data['HbA1c_level'], bins=9, ax=axes[0])
sns.boxplot(y=data['HbA1c_level'], ax=axes[1])
st.pyplot(fig)

# Histogramas e Boxplots para Nível de Glicose no Sangue
st.subheader("📊 Nível de Glicose no Sangue")
fig, axes = plt.subplots(1, 2, figsize=(15, 5))
sns.histplot(data['blood_glucose_level'], bins=9, ax=axes[0])
sns.boxplot(y=data['blood_glucose_level'], ax=axes[1])
st.pyplot(fig)

# Gráfico Total de Diabetes por Gênero
st.subheader("📊 Total de Diabetes por Gênero")
fig, ax = plt.subplots()
sns.countplot(data=data, x='diabetes', hue='gender', ax=ax)
st.pyplot(fig)

# 📌 Caminho padrão do dataset
caminho_padrao = "heart.csv"

# 📌 Título do aplicativo
st.title("📊 Análise de Dados do Coração")

# 📌 Verifica se o arquivo já existe no diretório
if os.path.exists(caminho_padrao):
    df = pd.read_csv(caminho_padrao)
    st.success("✅ Arquivo carregado automaticamente!")
else:
    st.warning("⚠️ Arquivo 'heart.csv' não encontrado. Faça o upload abaixo.")
    file = st.file_uploader("Carregue o arquivo 'heart.csv'", type="csv")
    
    if file:
        df = pd.read_csv(file)
        st.success("✅ Arquivo carregado com sucesso!")

# Se os dados foram carregados, continua a análise
if 'df' in locals():
    # 📌 Exibir primeiras linhas
    st.subheader("🔍 Primeiras Linhas do Dataset")
    st.dataframe(df.head())

    # 📌 Verificar dados ausentes
    st.subheader("📊 Mapa de Dados Ausentes")
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.heatmap(df.isnull(), cbar=False, cmap='coolwarm')
    st.pyplot(fig)

    # 📌 Informações do Dataset
    st.subheader("ℹ️ Informações do Dataset")
    buffer = df.info(buf=None)
    st.text(str(buffer))  

    # 📌 Histogramas interativos usando Plotly
    st.subheader("📊 Distribuição de Tipos de Dor no Peito")
    fig = px.histogram(df, x='ChestPainType', color='HeartDisease', template='plotly_dark')
    st.plotly_chart(fig)

    st.subheader("📊 Distribuição de Idades")
    fig = px.histogram(df, x='Age', color='HeartDisease', template='plotly_dark')
    st.plotly_chart(fig)

    st.subheader("📊 Distribuição por Sexo")
    fig = px.histogram(df, x='Sex', color='HeartDisease', template='plotly_dark')
    st.plotly_chart(fig)

    st.subheader("📊 Exercício e Angina")
    fig = px.histogram(df, x='Sex', color='ExerciseAngina', template='plotly_dark')
    st.plotly_chart(fig)

    # 📌 Pré-processamento: Convertendo colunas categóricas em numéricas
    label = LabelEncoder()
    categorias = ['Sex', 'RestingECG', 'ChestPainType', 'ExerciseAngina', 'ST_Slope']
    
    for coluna in categorias:
        df[coluna] = label.fit_transform(df[coluna])

    st.subheader("📊 Dados Após Pré-processamento")
    st.dataframe(df.head())
 
# 📌 Caminho padrão do dataset
caminho_padrao = "ObesityDataSet_raw_and_data_sinthetic.csv"

# 📌 Título do aplicativo
st.title("📊 Análise de Dados sobre Obesidade")

# 📌 Verifica se o arquivo já existe no diretório
if os.path.exists(caminho_padrao):
    df = pd.read_csv(caminho_padrao)
    st.success("✅ Arquivo carregado automaticamente!")
else:
    st.warning("⚠️ Arquivo não encontrado. Faça o upload abaixo.")
    file = st.file_uploader("Carregue o arquivo 'ObesityDataSet_raw_and_data_sinthetic.csv'", type="csv")
    
    if file:
        df = pd.read_csv(file)
        st.success("✅ Arquivo carregado com sucesso!")

# Se os dados foram carregados, continua a análise
if 'df' in locals():
    # 📌 Renomeando coluna
    df.rename(columns={"NObeyesdad": "NObesity"}, inplace=True)

    # 📌 Exibir primeiras linhas
    st.subheader("🔍 Primeiras Linhas do Dataset")
    st.dataframe(df.head())

    # 📌 Exibir número de observações e atributos
    st.write(f"📊 **Número de Observações:** {df.shape[0]}")
    st.write(f"📊 **Número de Atributos:** {df.shape[1]}")

    # 📌 Informações do Dataset
    st.subheader("ℹ️ Informações do Dataset")
    buffer = df.info(buf=None)
    st.text(str(buffer))  

    # 📌 Estatísticas descritivas
    st.subheader("📈 Estatísticas Descritivas")
    st.write(df.describe(include="all"))

    # 📌 Verificar dados ausentes
    st.subheader("📊 Mapa de Dados Ausentes")
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.heatmap(df.isnull(), cbar=False, cmap='coolwarm')
    st.pyplot(fig)

    # 📌 Identificando colunas numéricas e categóricas
    numerical_var = [var for var in df.columns if df[var].dtype != "O"]
    categorical_var = [var for var in df.columns if var not in numerical_var]

    # 📌 Histogramas das variáveis numéricas
    st.subheader("📊 Distribuição das Variáveis Numéricas")
    fig, axes = plt.subplots(nrows=5, ncols=2, figsize=(12, 20))
    axes = axes.flatten()
    for i, var in enumerate(numerical_var[:10]):
        sns.histplot(df[var], ax=axes[i])
        axes[i].set_title(var)
    plt.tight_layout()
    st.pyplot(fig)

    # 📌 Boxplots das variáveis numéricas
    st.subheader("📊 Boxplots das Variáveis Numéricas")
    fig, axes = plt.subplots(nrows=5, ncols=2, figsize=(12, 20))
    axes = axes.flatten()
    for i, var in enumerate(numerical_var[:10]):
        sns.boxplot(y=df[var], ax=axes[i])
        axes[i].set_title(var)
    plt.tight_layout()
    st.pyplot(fig)

    # 📌 Gráfico de Pizza - Tipos de Obesidade
    obesity_counts = df["NObesity"].value_counts().reset_index()
    obesity_counts.columns = ["ObesityType", "Count"]

    fig = px.pie(obesity_counts, names="ObesityType", values="Count", 
             title="Distribuição dos Tipos de Obesidade")


    # 📌 Gráficos de Barras para Variáveis Categóricas vs Obesidade
    st.subheader("📊 Comparação entre Variáveis Categóricas e Obesidade")
    for var in categorical_var[:9]:
        fig, ax = plt.subplots(figsize=(10, 4))
        sns.countplot(x="NObesity", hue=var, data=df, ax=ax)
        plt.title(f"{var} vs Obesity")
        st.pyplot(fig)

    # 📌 Gráficos de Barras para Variáveis Numéricas vs Obesidade
    st.subheader("📊 Relação entre Variáveis Numéricas e Obesidade")
    for var in numerical_var[:9]:
        fig, ax = plt.subplots(figsize=(10, 4))
        sns.barplot(x="NObesity", y=var, data=df, ax=ax)
        plt.title(f"{var} vs Obesity")
        st.pyplot(fig)

    # 📌 Matriz de Correlação
    st.subheader("📊 Matriz de Correlação entre Variáveis Numéricas")
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(df[numerical_var].corr(), annot=True, square=True, center=0, vmin=-1, vmax=1, cmap='BrBG', fmt='.2f', linewidths=5, annot_kws={"size": 7})
    st.pyplot(fig)   
# Título do aplicativo
st.title("📊 Análise Nutricional")

# Verificar se o arquivo existe no diretório
file_path = "nutrition.csv"
if os.path.exists(file_path):
    # Carregar os dados
    dfNutri = pd.read_csv(file_path)
    
    # Selecionar colunas relevantes
    dfNutri = dfNutri[['calories', 'total_fat', 'saturated_fat', 'fat', 'cholesterol', 'protein', 'carbohydrate', 'sugars', 'fiber', 'sodium', 'potassium', 'water']]
    
    # Converter colunas para valores numéricos removendo unidades
    for col in ['total_fat', 'saturated_fat', 'fat', 'protein', 'carbohydrate', 'sugars', 'fiber', 'water']:
        dfNutri[col] = dfNutri[col].astype(str).str.replace('g', '', regex=False).astype(float)
    
    for col in ['cholesterol', 'sodium', 'potassium']:
        dfNutri[col] = dfNutri[col].astype(str).str.replace('mg', '', regex=False).astype(float)
    
    # Remover valores nulos
    dfNutri = dfNutri.dropna()
    
    # Exibir as primeiras linhas do dataset
    st.subheader("🔍 Primeiras Linhas do Dataset")
    st.dataframe(dfNutri.head())
    
    # Exibir informações gerais
    st.subheader("ℹ️ Informações do Dataset")
    buffer = dfNutri.info(buf=None)
    st.text(buffer)
    
    # Exibir estatísticas descritivas
    st.subheader("📈 Estatísticas Descritivas")
    st.write(dfNutri.describe())
    
else:
    st.error("❌ O arquivo 'nutrition.csv' não foi encontrado no diretório.")
