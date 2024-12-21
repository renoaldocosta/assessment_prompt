import streamlit as st
import json
import time
import os
from dotenv import load_dotenv
from dataprep import (
    get_deputados, 
    create_pizza_chart, 
    generate_insights_about_chamber_of_deputies,
    get_expenses_by_deputy,
    chaining_generate_analisys,
    chaining_generate_analisys_json,
    chaining_load_analises,
    chaining_generate_insights,
    get_propositions,
    summarizer_chunk
    )
import google.generativeai as genai
import pandas as pd
from chaining_utils import chaining_remove_images, chaining_remove_analises


load_dotenv('../.env')
api_key = os.environ["GEMINI_KEY"]
model_flash = genai.GenerativeModel("gemini-1.5-flash")
model_pro = genai.GenerativeModel("gemini-1.5-pro")

# @st.cache_data
def load_parquet(path:str):
    return pd.read_parquet(path)

def run():
    st.title("Chat Application with LLM")

    
    

    with st.expander("## **3.A** - Colete e salve os dados dos deputados atuais da câmara no arquivo data/deputados.parquet através da url: url_base+/deputados"):
        col1 = st.columns(2)
        with col1[0]:
            if st.button("Coletar dados", use_container_width=True,type="primary"):
                try:
                    df_deputados = get_deputados()
                    st.write("Dados coletados com sucesso!")
                except Exception as e:
                    st.error(f"Erro ao coletar dados: {e}")
        with col1[1]:
            if st.button("Carregar dados", use_container_width=True):
                try:
                    df_deputados = load_parquet("../data/deputados.parquet")
                    st.write("Dados carregados com sucesso!")
                except Exception as e:
                    st.error(f"Erro ao carregar dados: {e}")
        try:
            if df_deputados is not None:
                st.write(df_deputados)
        except:
            pass
    with st.expander("## **3.B** - Executar prompt para criar o código que gere um gráfico de pizza com o total e o percentual de deputados de cada partido, salvo em 'docs/distribuicao_deputados.png"):
        if st.button("Executar prompt", use_container_width=True, type="primary"):
            try:
                codigo = create_pizza_chart()
                st.write("Código gerado com sucesso!")
            except Exception as e:
                st.error(f"Erro ao executar prompt: {e}")
            try:
                if codigo is not None:
                    st.code(codigo)
                    try:
                        st.image("../docs/distribuicao_deputados.png")
                    except Exception as e:
                        st.error(f"Erro ao exibir imagem: {e}")
            except Exception as e:
                st.error(f"Erro ao exibir código: {e}")
    with st.expander("**3.C** - Executar prompt utilizando os resultados da análise anterior (distribuição de deputados por partido) para gerar insights sobre a distribuição de partidos e como isso influencia a câmara.", expanded=False):
        if st.button("Executar prompt", use_container_width=True, type="primary", key="insights"):
            try:
                dicionario_insights = generate_insights_about_chamber_of_deputies()
                # st.write("Insights gerados com sucesso!")   
            except Exception as e:
                st.error(f"Erro ao executar prompt: {e}")
            try:
                if dicionario_insights is not None:
                    st.write(dicionario_insights)
                    # insigts = dicionario_insights['insights']
                    # # st.subheader(f"Insight gerados: {len(insigts)}")
                    # for insight in insigts:
                    #     st.markdown('- Tópico:', insight['topic'])
                    #     st.markdown('- Análise:', insight['analysis'])
                    #     st.write("\n\n")
            except Exception as e:
                st.error(f"Erro ao exibir insights: {e}")
    
    # Agrupe os dados de despesas por dia, deputado e tipo de despesa e salve num arquivo parquet (data/serie_despesas_diárias_deputados.parquet).
    with st.expander("## **4.A** - Agrupe os dados de despesas por dia, deputado e tipo de despesa e salve num arquivo parquet (data/serie_despesas_diárias_deputados.parquet)."):
        quantidade_deputados = st.selectbox("Quantidade de deputados",["Todos","Amostra"], index=1)
        if quantidade_deputados == "Todos":
            todos = True
        else:
            todos = False
            
        col1 = st.columns(2)
        with col1[0]:
            if st.button("Coletar dados", use_container_width=True, key="get_expenses",type="primary"):
                try:
                    df_expenses_agrupadas = get_expenses_by_deputy(todos)
                    st.info("Dados coletados com sucesso!")
                except Exception as e:
                    st.error(f"Erro ao coletar dados: {e}")
        with col1[1]:
            if st.button("Carregar dados", use_container_width=True, key="load_expenses"):
                try:
                    df_expenses_agrupadas = load_parquet("../data/serie_despesas_diárias_deputados.parquet")
                    st.info("Dados carregados com sucesso!")
                except Exception as e:
                    st.error(f"Erro ao carregar dados: {e}")
        try:
            if df_expenses_agrupadas is not None:
                st.write(df_expenses_agrupadas)
                st.write(df_expenses_agrupadas.columns)
        except:
            pass
    # 4B - Utilizando a técnica de prompt-chaining, crie um prompt que instrua o LLM a gerar um código python que analise os dados das despesas dos deputados. Peça para o LLM até 3 análises. Indique ao LLM quais dados estão disponíveis e o respectivo arquivo (salvo em a)) e execute as análises.
    with st.expander("## **4.B.1** - Utilizando a técnica de prompt-chaining, crie um prompt que instrua o LLM a gerar um código python que analise os dados das despesas dos deputados. Peça para o LLM até 3 análises. Indique ao LLM quais dados estão disponíveis e o respectivo arquivo (salvo em a)) e execute as análises."):
        if st.button("Executar prompt e Retornar Plano de Análises", use_container_width=True, key="chaining", type="primary"):
            try:
                analysis = chaining_generate_analisys(model_flash)
                st.session_state["analysis"] = analysis
                st.write("Análises geradas com sucesso!")
            except Exception as e:
                st.error(f"Erro ao executar prompt: {e}")
            try:
                if analysis is not None:
                    st.write(analysis)
            except Exception as e:
                st.error(f"Erro ao exibir análises: {e}")
    # 4B.2 - Carregar os dados das análises geradas e salve em um arquivo json (data/analise1.json, data/analise2.json, data/analise3.json).
    with st.expander("## **4.B.2** - Carregar os dados das análises geradas e salve em um arquivo json (data/analise1.json, data/analise2.json, data/analise3.json)."):
        if st.button("Executar prompt e Retornar Codigos e Jsons com informações", use_container_width=True, key="chaining_json", type="primary"):
            chaining_remove_images()
            chaining_remove_analises()
            if 'analysis' in st.session_state:
                try:
                    
                    list_codes = chaining_generate_analisys_json(st.session_state["analysis"], model_pro)
                    st.write("Análises salvas com sucesso!")
                    try:
                        with st.subheader("Códigos gerados"):
                            if list_codes is not None:
                                for code in list_codes:
                                    st.code(code)
                    except Exception as e:
                        st.error(f"Erro ao exibir Códigos gerados: {e}")
                except Exception as e:
                    st.error(f"Erro ao executar prompt: {e}")

        try:
            with st.subheader("Jsons gerados"):
                    try:
                        for n in range(1,4):
                            st.write(json.load(open(f"../data/analise{n}.json")))
                    except Exception as e:
                        st.error(f"Erro ao carregar dados (Jsons gerados): {e}")
        except:
            pass
        try:
            from PIL import Image
            # Defina o diretório onde as imagens estão localizadas
            image_folder = "../data/images"

            # Lista todos os arquivos PNG no diretório
            image_files = [f for f in os.listdir(image_folder) if f.endswith('.png')]

            # Exibe as imagens no Streamlit
            for image_file in image_files:
                image_path = os.path.join(image_folder, image_file)
                image = Image.open(image_path)
                st.image(image, caption=image_file)
        except Exception as e:  
            st.error(f"Erro ao exibir imagens: {e}")
    # 4C - Utilize os resultados das 3 análises para criar um prompt usando a técnica de Generated Knowledge para instruir o LLM a gerar insights. Salve o resultado como um JSON (data/insights_despesas_deputados.json).
    with st.expander("## **4.C** - Utilize os resultados das 3 análises para criar um prompt usando a técnica de Generated Knowledge para instruir o LLM a gerar insights. Salve o resultado como um JSON (data/insights_despesas_deputados.json)."):
        if st.button("Executar prompt e Retornar Insights", use_container_width=True, key="insights_despesas", type="primary"):
            try:
                analise1, analise2, analise3 = chaining_load_analises()
                today = time.strftime("%Y-%m-%d %H:%M:%S")
                insights = chaining_generate_insights(analise1, analise2, analise3, today, model_flash)
                st.write("Insights gerados com sucesso!")
                try:
                    if insights is not None:
                        st.write(insights)
                except Exception as e:
                    st.error(f"Erro ao exibir insights: {e}")
            except Exception as e:
                st.error(f"Erro ao executar prompt: {e}")
    
    # 5A - Coletar um total de 10 proposiçoes por tema e salvar em data/proposicoes_deputados.parquet
    with st.expander("## **5.A** - Coletar um total de 10 proposiçoes por tema e salvar em data/proposicoes_deputados.parquet"):
        col1 = st.columns(2)
        with col1[0]:
            if st.button("Coletar dados", use_container_width=True, key="get_proposicoes",type="primary"):
                try:
                    df_proposicoes = get_propositions()
                    st.session_state["df_proposicoes"] = df_proposicoes
                    st.info("Dados coletados com sucesso!")
                except Exception as e:
                    st.error(f"Erro ao coletar dados: {e}")
        with col1[1]:
            if st.button("Carregar dados", use_container_width=True, key="load_proposicoes"):
                try:
                    df_proposicoes = load_parquet("../data/proposicoes_deputados.parquet")
                    st.info("Dados carregados com sucesso!")
                except Exception as e:
                    st.error(f"Erro ao carregar dados: {e}")
        try:
            if df_proposicoes is not None:
                df_proposicoes = df_proposicoes.reset_index(drop=True)
                st.write(df_proposicoes)
        except:
            pass
    # 5B - Utilize a sumarização por chunks para resumir as proposições tramitadas no período de referência. Avalie a resposta e salve-a em data/sumarizacao_proposicoes.json
    with st.expander("## **5.B** - Utilize a sumarização por chunks para resumir as proposições tramitadas no período de referência. Avalie a resposta e salve-a em data/sumarizacao_proposicoes.json"):
        if 'df_proposicoes' in st.session_state:
            if st.button("Executar prompt e Retornar Sumarização", use_container_width=True, key="sumarizacao", type="primary"):
                try:
                    summarizer_chunks, summarizer_chunk_summaries, propsition_summary = summarizer_chunk(st.session_state["df_proposicoes"])
                    st.write("Sumarização gerada com sucesso!")
                    try:
                        if propsition_summary is not None:
                            tab3 = st.tabs(["Chunks","Sumarização por Chunks","Sumarização"])
                            with tab3[0]:
                                st.write(summarizer_chunks)
                            with tab3[1]:
                                st.write(summarizer_chunk_summaries)
                            with tab3[2]:
                                if 'theme' in propsition_summary:
                                    st.write(f"Tema: {propsition_summary['theme']}")
                                st.write(f"Resumo: {propsition_summary['summary']}")
                    except Exception as e:
                        st.error(f"Erro ao exibir sumarização: {e}")
                except Exception as e:
                    st.error(f"Erro ao executar prompt: {e}")
                    
                    
                    



if __name__ == "__main__":
    run()

