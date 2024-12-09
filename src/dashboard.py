import streamlit as st
import yaml
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

st.set_page_config(page_title="Câmara dos Deputados", layout="centered")

tab1, tab2, tab3 = st.tabs(["Overview", "Despesas", "Proposições"])

with tab1:
    st.title("Câmara dos Deputados")
    
    # Carregar arquivo YAML
    with open("../data/config.yaml", "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    
    # Exibir texto sumarizado
    st.write(config["overview_summary"])
    
    # Exibir gráfico
    st.image("../docs/distribuicao_deputados.png", use_container_width=True)
    
    # Carregar e exibir insights
    with open("../data/insights_distribuicao_deputados.json", "r", encoding="utf-8") as file:
        insights_data = json.load(file)
        
    for insight in insights_data["insights"]:
        st.subheader(insight["topic"])
        st.write(insight["analysis"])

with tab2:
    st.title("Despesas dos Deputados")
    
    # Carregar e exibir insights sobre despesas
    with open("../data/insights_despesas_deputados.json", "r", encoding="utf-8") as file:
        insights_despesas = json.load(file)
    
    for insight in insights_despesas["insights"]:
        st.subheader(insight["Topic"])
        st.write(insight["Insight"])
    
    # Carregar dados dos deputados e criar selectbox
    df_deputados = pd.read_parquet("../data/deputados.parquet")
    deputado_selecionado = st.selectbox(
        "Selecione um deputado:",
        options=list(zip(df_deputados['id'], df_deputados['nome'])),
        format_func=lambda x: x[1]
    )
    
    # Carregar e plotar série temporal de despesas
    df_despesas = pd.read_parquet("../data/serie_despesas_diárias_deputados.parquet")
    
    # Filtrar dados do deputado selecionado
    df_desp_deputado = df_despesas[df_despesas['id'] == deputado_selecionado[0]]
    
    # Criar gráfico
    plt.figure(figsize=(12, 6))
    sns.barplot(data=df_desp_deputado, x='dia', y='valorLiquido', hue='tipo_despesa')
    plt.title(f'Despesas do Deputado {deputado_selecionado[1]}')
    plt.xlabel('Data')
    plt.ylabel('Valor (R$)')
    plt.xticks(rotation=90)
    plt.grid(True)
    plt.tight_layout()
    
    # Exibir gráfico no Streamlit
    st.pyplot(plt)

with tab3:
    st.title("Proposições")
    
    # Carregar e exibir tabela de proposições
    df_proposicoes = pd.read_parquet("../data/proposicoes_deputados.parquet")
    st.dataframe(df_proposicoes)
    
    # Carregar e exibir resumo das proposições
    with open("../data/sumarizacao_proposicoes.json", "r", encoding="utf-8") as file:
        sumarizacao = json.load(file)
    
    st.subheader("Tema")
    st.write(sumarizacao["theme"])
    
    st.subheader("Resumo")
    st.write(sumarizacao["summary"])