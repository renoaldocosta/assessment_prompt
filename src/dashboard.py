import streamlit as st
import yaml
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

from dataprep import (
    load_and_process_data,
    return_questions_from_one_question
    )

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
    
    st.subheader("Chat Interface")
    
    perguntas = ['Qual é o partido político com mais deputados na câmara?',
                'Qual é o deputado com mais despesas na câmara?',
                'Qual é o tipo de despesa mais declarada pelos deputados da câmara?',
                'Quais são as informações mais relevantes sobre as proposições que falam de Economia?',
                "Quais são as informações mais relevantes sobre as proposições que falam de 'Ciência, Tecnologia e Inovação'?"
                ]
    # st.write(perguntas)
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    messages = st.container(height=300)
    
    def atualizar_mensagens(messages):
        # Contêiner para exibir mensagens
        with messages:
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
    
    def gerar_resposta(prompt):
        import time
        time.sleep(2)
        return 'Você disse: '+ prompt
    
    if prompt := st.chat_input("Escreva sua pergunta:"):
        
        atualizar_mensagens(messages)
        
        with messages.chat_message("user"):
            st.markdown(prompt)
            
        response = gerar_resposta(prompt)
        
        with messages.chat_message("assistant"):
            st.markdown(response)
            
        
        
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.session_state.messages.append({"role": "assistant", "content": response})
        
        
        
    st.title('\n')
    st.title('\n')
    st.title('\n')
    st.title('\n')
        
        
        
        
        
        
        
        
        
        
    
    parquet_path = '../data/df_informations.parquet'
    # model_name = 'neuralmind/bert-base-portuguese-cased'
    model_name = 'all-MiniLM-L6-v2'
    llm_model_dir = '../data/bertimbau/'
    k = 20 # número de resultados mais próximos
    candidate_count = 5 # número de respostas geradas pelo Gemini Flash
    

    perguntas = ['Qual é o partido político com mais deputados na câmara?',
                'Qual é o deputado com mais despesas na câmara?',
                'Qual é o tipo de despesa mais declarada pelos deputados da câmara?',
                'Quais são as informações mais relevantes sobre as proposições que falam de Economia?',
                "Quais são as informações mais relevantes sobre as proposições que falam de 'Ciência, Tecnologia e Inovação'?"
                ]
    
    question = st.selectbox('Escolha uma pergunta:', perguntas, index=0)   
    if st.button('Buscar Resposta'):
        if 'index' not in st.session_state or 'texts' not in st.session_state or 'embedding_model' not in st.session_state:
            try:
                with st.status('Criando Base de dados...', expanded=True) as status:
                    texts, index, embedding_model = load_and_process_data(parquet_path, model_name, llm_model_dir)
                    st.session_state['texts'] = texts
                    st.session_state['index'] = index
                    st.session_state['embedding_model'] = embedding_model
                    status.update(
                    label="Base de Dados criada com sucesso!", state="complete", expanded=False
                )
            except Exception as e:
                print(f'################# Erro ao carregar e processar os dados {e}')
        else:
            texts = st.session_state['texts']
            index = st.session_state['index']
            embedding_model = st.session_state['embedding_model']
            st.toast("Index carregado com sucesso!")

        try:
            with st.status('Obtendo Resposta...', expanded=True) as status:
                questions, question_answers, response, responses, traducao = return_questions_from_one_question(question, texts, index, embedding_model, k, candidate_count)
                st.session_state['resposta'] = traducao 
                status.update(
                label="Resposta obtida!", state="complete", expanded=False
            )  
            st.write(f'**Resposta**: {st.session_state["resposta"]}')
            
        except Exception as e:
            status.update(
                label="Falha!", state="error", expanded=False
            ) 
            print(f'################# Erro ao retornar as perguntas da questão {e}')
            
    
    
    
    
    
    
    st.title('\n')
    st.title('\n')
    st.title('\n')
    st.title('\n')
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