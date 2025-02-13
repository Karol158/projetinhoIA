
import streamlit as st
import pandas as pd
import plotly.express as px
import os
from sklearn.preprocessing import LabelEncoder
DATA_PATH = "food_coded.csv"  # Caminho fixo do arquivo

def carregar_dados():
    try:
        df = pd.read_csv(DATA_PATH)
        return df
    except FileNotFoundError:
        st.error("Erro: O arquivo 'food_coded.csv' não foi encontrado no diretório.")
        return None

st.title("Análise de Dados Categóricos")
df = carregar_dados()
if df is not None:
    object_columns = df.select_dtypes(include=['object']).columns
    st.subheader("Colunas Categóricas")
    st.write(df[object_columns].head())
    
    coluna_selecionada = st.selectbox("Selecione uma coluna categórica", object_columns)
    
    if coluna_selecionada:
        st.subheader(f"Distribuição de valores - {coluna_selecionada}")
        st.write(df[coluna_selecionada].value_counts())
        
        # Criar DataFrame com a contagem de valores
        contagem_valores = df[coluna_selecionada].value_counts().reset_index()
        contagem_valores.columns = [coluna_selecionada, "count"]  # Renomeia colunas corretamente

        # Criar gráfico de barras
        fig = px.bar(contagem_valores, 
                     x=coluna_selecionada, 
                     y="count", 
                     title=f"Distribuição de {coluna_selecionada}",
                     labels={coluna_selecionada: "Categoria", "count": "Frequência"})

        st.plotly_chart(fig)  # Exibir gráfico no Streamlit

# Título da aplicação para Diabetes
diabetes_csv = "diabetes_prediction_dataset.csv"
st.title("📊 Análise de Dados de Diabetes")

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
        st.stop()

st.subheader("📊 Contagem por Diabetes")
fig = px.histogram(data, x='diabetes', title="Contagem de Diabetes", color='diabetes')
st.plotly_chart(fig)

st.subheader("📊 Contagem por Gênero")
fig = px.histogram(data, x='gender', title="Distribuição de Gênero", color='gender')
st.plotly_chart(fig)

st.subheader("📊 Distribuição da Idade")
fig = px.histogram(data, x='age', title="Distribuição da Idade", nbins=20)
st.plotly_chart(fig)

st.subheader("📊 Contagem de Pacientes com Hipertensão")
fig = px.histogram(data, x='hypertension', title="Distribuição de Hipertensão", color='hypertension')
st.plotly_chart(fig)

st.subheader("📊 Contagem de Pacientes com Problemas Cardíacos")
fig = px.histogram(data, x='heart_disease', title="Distribuição de Problemas Cardíacos", color='heart_disease')
st.plotly_chart(fig)

st.subheader("📊 Contagem por Histórico de Tabagismo")
fig = px.histogram(data, x='smoking_history', title="Histórico de Tabagismo", color='smoking_history')
st.plotly_chart(fig)

st.subheader("📊 Nível de HbA1c")
fig = px.histogram(data, x='HbA1c_level', title="Distribuição do Nível de HbA1c", nbins=20)
st.plotly_chart(fig)

st.subheader("📊 Nível de Glicose no Sangue")
fig = px.histogram(data, x='blood_glucose_level', title="Distribuição do Nível de Glicose", nbins=20)
st.plotly_chart(fig)

st.subheader("📊 Total de Diabetes por Gênero")
fig = px.histogram(data, x='diabetes', color='gender', title="Diabetes por Gênero")
st.plotly_chart(fig)

# 📌 Caminho padrão do arquivo
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

# 📌 Se os dados foram carregados, continua a análise
if 'df' in globals() and df is not None:
    # 📌 Exibir primeiras linhas
    st.subheader("🔍 Primeiras Linhas do Dataset")
    st.dataframe(df.head())

    # 📌 Verificar dados ausentes usando um heatmap interativo
    st.subheader("📊 Mapa de Dados Ausentes")
    missing_data = df.isnull().astype(int)  # Converte valores booleanos para 0 e 1
    fig = px.imshow(missing_data, color_continuous_scale='reds', title="Mapa de Dados Ausentes")
    st.plotly_chart(fig)

    # 📌 Informações do Dataset
    st.subheader("ℹ️ Informações do Dataset")
    buffer = df.dtypes.astype(str)  # Converte tipos de dados em string para exibição
    st.text("\n".join(f"{col}: {dtype}" for col, dtype in buffer.items()))

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

# 📌 Se os dados foram carregados, continua a análise
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
    buffer = df.dtypes.astype(str)
    st.text("\n".join(f"{col}: {dtype}" for col, dtype in buffer.items()))

    # 📌 Estatísticas descritivas
    st.subheader("📈 Estatísticas Descritivas")
    st.write(df.describe(include="all"))

    # 📌 Verificar dados ausentes (Mapa de calor)
    st.subheader("📊 Mapa de Dados Ausentes")
    missing_data = df.isnull().astype(int)
    fig = px.imshow(missing_data, color_continuous_scale="reds", title="Mapa de Dados Ausentes")
    st.plotly_chart(fig)

    # 📌 Identificando colunas numéricas e categóricas
    numerical_var = [var for var in df.columns if df[var].dtype != "O"]
    categorical_var = [var for var in df.columns if var not in numerical_var]

    # 📌 Histogramas das variáveis numéricas
    st.subheader("📊 Distribuição das Variáveis Numéricas")
    for var in numerical_var[:10]:
        fig = px.histogram(df, x=var, title=f"Distribuição de {var}", template="plotly_dark")
        st.plotly_chart(fig)

    # 📌 Boxplots das variáveis numéricas
    st.subheader("📊 Boxplots das Variáveis Numéricas")
    for var in numerical_var[:10]:
        fig = px.box(df, y=var, title=f"Boxplot de {var}", template="plotly_dark")
        st.plotly_chart(fig)

    # 📌 Gráfico de Pizza - Tipos de Obesidade
    obesity_counts = df["NObesity"].value_counts().reset_index()
    obesity_counts.columns = ["ObesityType", "Count"]
    fig = px.pie(obesity_counts, names="ObesityType", values="Count", title="Distribuição dos Tipos de Obesidade")
    st.plotly_chart(fig)

    # 📌 Gráficos de Barras para Variáveis Categóricas vs Obesidade
    st.subheader("📊 Comparação entre Variáveis Categóricas e Obesidade")
    for var in categorical_var[:9]:
        fig = px.bar(df, x="NObesity", color=var, title=f"{var} vs Obesity", barmode="group", template="plotly_dark")
        st.plotly_chart(fig)

    # 📌 Gráficos de Barras para Variáveis Numéricas vs Obesidade
    st.subheader("📊 Relação entre Variáveis Numéricas e Obesidade")
    for var in numerical_var[:9]:
        fig = px.bar(df, x="NObesity", y=var, title=f"{var} vs Obesity", template="plotly_dark")
        st.plotly_chart(fig)

    # 📌 Matriz de Correlação
    st.subheader("📊 Matriz de Correlação entre Variáveis Numéricas")
    fig = px.imshow(df[numerical_var].corr(), text_auto=True, color_continuous_scale="RdBu_r", title="Matriz de Correlação")
    st.plotly_chart(fig)
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
