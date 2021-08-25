

#!pip install -q stramlit & "C:/Users/kenne/anaconda3/python.exe" "c:/projetodsia/exemplodedeploy.py"

import pandas as pd 
import streamlit as st 
import plotly.express as px 
from sklearn.ensemble import RandomForestRegressor

@st.cache
def get_data():
    return pd.read_csv("c:/projetodsia/data.csv")

def train_model():
    data = get_data()
    x = data.drop("MEDV", axis=1)
    y = data["MEDV"]
    rf_regressor = RandomForestRegressor()
    rf_regressor.fit(x, y)
    return rf_regressor 

data = get_data()

model = train_model() 

st.title("Data App - prevendo valores de imovéis")

st.markdown("Este é um data app utilizado para exibir a solução de machine learning para o problema de predição de valores de imóveis de Boston")

st.subheader("Selecionando apenas um pequeno conjnto de atributos")

defaultcols = ["RM","PTRATIO","LSTAT","MEDV"]

cols = st.multiselect("Atributos", data.columns.tolist(), default=defaultcols)

st.dataframe(data[cols].head(10))


st.subheader("Distribuição de imóveis por preço")

faixa_valores = st.slider("Faixa de preço", float(data.MEDV.min()), 150.,(10.0, 100.0))

dados = data[data['MEDV'].between(left=faixa_valores[0],right=faixa_valores[1])]

f = px.histogram(dados, x="MEDV", nbins=100,title="Distribição de preços")
f.update_xaxes(title="MEDV")
f.update_yaxes(title="Total Imóveis")
st.plotly_chart(f)

st.sidebar.subheader("Defina os atributos do imóvel para a predição")

crim = st.sidebar.number_input("Taxa de criminalidade", value=data.CRIM.mean())
indus = st.sidebar.number_input("Proporção de hectares de negócio", value=data.CRIM.mean())
chas = st.sidebar.selectbox("Faz limite com o rio ?", ("Sim","Não"))

chas = 1 if chas == "Sim" else 0

nox = st.sidebar.number_input("Concentração de óxido nítrico", value=data.NOX.mean())

rm = st.sidebar.number_input("Número de quartos", value=1)

ptratio = st.sidebar.number_input("Indice de alunos para professores", value=data.PTRATIO.mean())

b = st.sidebar.number_input("Proporção de pessoas com descendencia afro-americana", value=data.B.mean())

lstat = st.sidebar.number_input("Porcentagem de status baixo", value=data.LSTAT.mean())

btn_predict = st.sidebar.button("Realizar Predição")

if btn_predict:
    result = model.predict([[crim,indus,chas,nox,rm,ptratio,b,lstat]])
    st.subheader("O valor previsto para o imóvel é:")
    result = "US $ "+str(round(result[0]*10,2))
    st.write(result)