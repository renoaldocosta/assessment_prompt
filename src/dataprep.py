import requests
import pandas as pd
import numpy as np
import json
import os
import google.generativeai as genai
from dotenv import load_dotenv
import yaml
from tqdm import tqdm
from google.generativeai.types import HarmCategory, HarmBlockThreshold



camara_base_url = 'https://dadosabertos.camara.leg.br/api/v2/'
genai.configure(api_key=os.environ["GEMINI_KEY"])
model_flash = genai.GenerativeModel("gemini-1.5-flash")
model_pro = genai.GenerativeModel("gemini-1.5-pro")

def load_parquet(path:str):
    return pd.read_parquet(path)

def get_deputados():
    url = f'{camara_base_url}/deputados?dataInicio=2024-08-01&dataFim=2024-08-30&ordem=ASC&ordenarPor=nome'
    response = requests.get(url, timeout=30)
    if not response.ok: # != 200:
        raise Exception('Nao foi possivel recuperar os dados')

    df_deputados = pd.DataFrame().from_dict(json.loads(response.text)['dados'])
    df_deputados.to_parquet('../data/deputados.parquet')
    
    return df_deputados

def create_pizza_chart():
    prompt_start = """
    You are a Python programmer with expertise in data analysis. Your task is to create a Python script that performs the following actions:

    1. Load a dataset stored in a Parquet file located at '../data/deputados.parquet'. The dataset contains the following columns:
    ['id', 'uri', 'nome', 'siglaPartido', 'uriPartido', 'siglaUf', 'idLegislatura', 'urlFoto', 'email'].

    2. Use the column 'siglaPartido' to calculate:
    - The total number of deputies (rows) for each political party.
    - The percentage representation of each party.

    3. Save this information as a JSON file in the directory '../data' with the filename 'distribuicao_deputados.json'. 
    - The JSON should have the structure:
        [
            {
                "party": "Party Name",
                "total_deputies": TotalNumber,
                "percentage": PercentageValue
            }
        ]

    4. Use the JSON file to generate a pie chart that displays the distribution of deputies per party.
    - Use json.loads() to parse the JSON data.
    - The pie chart should display each party as a slice.
    - Each slice should show both the total count and percentage of deputies for the corresponding party.
    - Add a title, labels, and a legend to the plot.
    - Save the pie chart as an image file in the directory '../docs' with the filename 'distribuicao_deputados.png'.

    Output only the code, no need for explanations.
    """

    genai.configure(api_key=os.environ["GEMINI_KEY"])
    model = genai.GenerativeModel("gemini-1.5-pro")
    response = model.generate_content(prompt_start)
    analysis_code = response.text.replace("```python\n", '').replace("\n```", '')
    exec(analysis_code)
    return analysis_code


def generate_insights_about_chamber_of_deputies():
    
    with open('../data/config.yaml', 'r') as f:
        config = yaml.safe_load(f)['overview_summary']
        
    with open('../data/distribuicao_deputados.json', 'r') as f:
        data_json = json.load(f)

    # Escapar as chaves no f-string
    prompt_start = f"""
        You are a political analyst specializing in legislative systems. 
        Your task is to analyze the distribution of deputies among political parties in the Brazilian Chamber of Deputies based on the following data:

        JSON data:
        {data_json}

        Additionally, consider the following summary of the Chamber's role:
        {config}

        Using this information:
        1. Identify trends in the distribution of deputies across parties.
        2. Analyze the implications of party dominance on legislative negotiations.
        3. Explain how the presence of smaller parties affects coalition building and power balance.

        Generate a structured output in JSON format, where each insight includes:
        
        {{
        "insights": [
            {{
            "topic": "Dominance of Major Parties",
            "analysis": "Key Insight: The PARTY dominates the Chamber with XX% of the deputies. 
            This gives them a significant advantage in legislative negotiations."
            }},
            {{
            "topic": "Influence of Small Parties",
            "analysis": "Key Insight: Smaller parties, like Rede and S.Part., hold marginal influence. However, they can act as tie-breakers in close votes."
            }}
        ]
        }}
        Output only the JSON file, no need for explanations.
    """

    genai.configure(api_key=os.environ["GEMINI_KEY"])
    model = genai.GenerativeModel("gemini-1.5-pro")
    response = model.generate_content(prompt_start)
    dict_insights = json.loads(response.text.replace("```json\n",'').replace("\n```",''))
    try:
        with open('../data/insights_distribuicao_deputados.json', 'w') as f:
            json.dump(dict_insights, f)
    except Exception as e:
        print(e)
    return dict_insights


def get_expenses_by_deputy(todos=False):
    df_deputados = pd.read_parquet("../data/deputados.parquet")
    if not todos:
        ids_deputados = df_deputados['id'].unique()[0:5]
    else:
        ids_deputados = df_deputados['id'].unique()
    list_expenses = []
    anoDespesa = '2024'
    maxItens = '100' # somente suporta ate 100
    mes = '08'
    for id in tqdm(ids_deputados):
        url = f'{camara_base_url}/deputados/{id}/despesas'
        params = {
            'ano': anoDespesa,
            'mes': mes,
            'itens': maxItens,
        }
        # Execucao da primeira pagina de resultados.
        response = requests.get(url, params, timeout=30)
        df_resp = pd.DataFrame().from_dict(json.loads(response.text)['dados'])
        df_resp['id'] = id
        list_expenses.append(df_resp)
        # Link para proxima pagina
        df_links = pd.DataFrame().from_dict(json.loads(response.text)['links'])
        df_links = df_links.set_index('rel').href
        
        while 'next' in df_links.index:
            response = requests.get(df_links['next'], timeout=30)
            df_resp = pd.DataFrame().from_dict(json.loads(response.text)['dados'])
            df_resp['id'] = id
            list_expenses.append(df_resp)
            # Link para proxima pagina
            df_links = pd.DataFrame().from_dict(json.loads(response.text)['links'])
            df_links = df_links.set_index('rel').href
            
    df_expenses = pd.concat(list_expenses)

    # Merge para trazer as informacoes de sigla e afins do deputado
    df_expenses = df_expenses.merge(df_deputados, on=['id'])

    df_expenses["dataDocumento"] = pd.to_datetime(df_expenses["dataDocumento"], errors='coerce')
    
    df_expenses = df_expenses[
        (df_expenses['dataDocumento'] >= '2024-08-01') & 
        (df_expenses['dataDocumento'] <= '2024-08-30')
    ]

    # Agrupar por dia, deputado e tipo de despesa
    df_expenses_agrupadas = df_expenses.groupby(
        [df_expenses["dataDocumento"].dt.date, "id","nome", "tipoDespesa"]
    ).agg({
        "valorDocumento": "sum",
        "valorLiquido": "sum",
        "valorGlosa": "sum"
    }).reset_index()

    # Renomear as colunas
    df_expenses_agrupadas.rename(columns={
        "dataDocumento": "dia",
        "idDeputado": "deputado",
        "tipoDespesa": "tipo_despesa"
    }, inplace=True)
    
    
    # Salvar em arquivo Parquet
    output_path = "../data/serie_despesas_diárias_deputados.parquet"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_expenses_agrupadas.to_parquet(output_path, index=False)
    print(f"Arquivo salvo em {output_path}")
    return df_expenses_agrupadas


def chaining_generate_analisys(model):
    prompt_start = """
    You are a Python data scientist specializing in data analysis and visualization. 
    Your task is to analyze a dataset containing daily expense records of Brazilian deputies. 
    The dataset is saved in a Parquet file located at "../data/serie_despesas_diárias_deputados.parquet". 
    The file contains the following columns:

    - dia: The date of the expense.
    - id: Unique identifier of the deputy.
    - nome: Name of the deputy.
    - tipo_despesa: The category/type of the expense.
    - valorDocumento: The documented value of the expense.
    - valorLiquido: The net value of the expense (after deductions or adjustments).
    - valorGlosa: The amount of expense denied or glossed.

    Here is a sample of the data:
    dia id  nome    tipo_despesa    valorDocumento  valorLiquido    valorGlosa
    2024-08-20	92699	Fernando Monteiro	LOCAÇÃO OU FRETAMENTO DE VEÍCULOS AUTOMOTORES	16000	12713	3287
    2024-08-07	220570	Ismael Alexandrino	PASSAGEM AÉREA - REEMBOLSO	15700	12713	2987

    Suggest 3 descriptive analyses that can be performed with this dataset, focusing on:
    1. Trends or patterns in expenses over time.
    2. Categories of expenses with the highest and lowest values.
    3. Deputies with the most and least expenses, including net values and glossed amounts.
    4. Do not use kaleido

    Format the output as a JSON with the following structure:
    {[
        {
        "Name": "Name of the analysis",
        "Objective": "Purpose of the analysis",
        "Method": "How to implement the analysis"
        }
    ]
    }
    """

    response = model.generate_content(prompt_start)
    analysis = json.loads(response.text.replace("```json\n",'').replace("\n```",''))
    return analysis


def chaining_generate_analisys_json(analysis, model):
    list_codes = []
    for x, analise in enumerate(analysis):

        json_analise = json.loads(json.dumps(analysis[x]))
        json_analise['Name']

        prompt_start = f"""
        You are a Python data scientist specializing in data analysis and visualization. 
        Your task is to analyze a dataset containing daily expense records of Brazilian deputies. 
        The dataset is saved in a Parquet file located at "../data/serie_despesas_diárias_deputados.parquet". 
        The file contains the following columns:

        - dia: The date of the expense.
        - id: Unique identifier of the deputy.
        - nome: Name of the deputy.
        - tipo_despesa: The category/type of the expense.
        - valorDocumento: The documented value of the expense.
        - valorLiquido: The net value of the expense (after deductions or adjustments).
        - valorGlosa: The amount of expense denied or glossed.

        Here is a sample of the data:
        dia id  nome    tipo_despesa    valorDocumento  valorLiquido    valorGlosa
        2024-08-20	92699	Fernando Monteiro	LOCAÇÃO OU FRETAMENTO DE VEÍCULOS AUTOMOTORES	16000	12713	3287
        2024-08-07	220570	Ismael Alexandrino	PASSAGEM AÉREA - REEMBOLSO	15700	12713	2987

        Implement the analysis described below in python.
        Output only the code, no need for explanations.
        ## ANALYSIS
        {analysis[x]}

        Task: Implement the analysis described above in Python. Follow these steps:

        1. Load the data from the provided Parquet file.
        2. Implement the "{json_analise['Name']}" method.
        3. Save the results of the analysis, including any statistical findings, as a JSON file at "../data/analise{x+1}.json".
        4. if you need to save images, save them in the directory "../data/images".
        5. All graphs should be have a title, labels, and a legend.
        6. Do not use kaleido

        Format the output as a JSON with the following structure:
        {[
        {
            "Name": "Name of the analysis",
            "Findings": "The resolts of the analysis",
        }
        ]
        }
        """

        # Definir a chave de API do Gemini (use a chave fornecida pela sua conta)
        response = model.generate_content(prompt_start)
        analysis_code = response.text.replace("```python\n",'').replace("\n```",'')
        list_codes.append(analysis_code)
        exec(analysis_code)
    return list_codes


def chaining_load_analises():
    with open("../data/analise1.json", "r") as file:
        analise1 = json.load(file)

    with open("../data/analise2.json", "r") as file:
        analise2 = json.load(file)

    with open("../data/analise3.json", "r") as file:
        analise3 = json.load(file)
    
    return analise1, analise2, analise3



def chaining_generate_insights(analise1, analise2, analise3, today, model):

    prompt_start = f"""
    You are a Python data scientist specializing in data analysis and interpretation. 
    You have been provided with the results of three descriptive analyses on the daily expense records of Brazilian deputies. 
    Your task is to generate actionable insights based on these results. The goal is to synthesize meaningful conclusions that highlight patterns, trends, and significant findings related to the deputies' expenses.

    ### Results of the Analyses:

    1. **{analise1[0]['Name']}**
    {analise1[0]['Findings']}

    2. **{analise2[0]['Name']}**
    {analise2[0]['Findings']}

    3. **{analise3[0]['Name']}**
    {analise3[0]['Findings']}


    # Task:
    Using the results of these analyses:
    1. Generate **3 to 5 actionable insights** that provide meaningful conclusions about the deputies' spending habits, patterns, or anomalies.
    2. Ensure the insights are concise, clear, and data-driven.
    3. Format the output as a JSON file

    Here is a sample of the data:
    dia id  nome    tipo_despesa    valorDocumento  valorLiquido    valorGlosa
    2024-08-20	92699	Fernando Monteiro	LOCAÇÃO OU FRETAMENTO DE VEÍCULOS AUTOMOTORES	16000	12713	3287
    2024-08-07	220570	Ismael Alexandrino	PASSAGEM AÉREA - REEMBOLSO	15700	12713	2987

    Suggest 3 descriptive analyses that can be performed with this dataset, focusing on:
    1. Trends or patterns in expenses over time.
    2. Categories of expenses with the highest and lowest values.
    3. Deputies with the most and least expenses, including net values and glossed amounts.

    Format the output as a JSON with the following structure:
    ```json
    {{
        "insights": [
        {{
            "Topic": "Brief title for the insight",
            "Insight": "The main finding or conclusion based on the analysis.",
            }},
        {{
            "Topic": "Brief title for the insight",
            "Insight": "The main finding or conclusion based on the analysis.",
            }},
        {{
            "Topic": "Brief title for the insight",
            "Insight": "The main finding or conclusion based on the analysis.",
            }}
        }}
        ],
        "Date":"{{today}}"
    }}

    Output only the JSON file, no need for explanations.
    """

    response = model.generate_content(prompt_start)
    insights = json.loads(response.text.replace("```json\n",'').replace("\n```",''))
    with open("../data/insights_despesas_deputados.json", "w",encoding="UTF-8") as file:
        json.dump(insights, file)
    return insights


# 5A - Coletar um total de 10 proposiçoes por tema e salvar em data/proposicoes_deputados.parquet
######## ==============================================================
## Questão 5A - Coletar um total de 10 proposiçoes por tema e salvar em data/proposicoes_deputados.parquet
## 
##
##

def get_propositions():
    camara_base_url = 'https://dadosabertos.camara.leg.br/api/v2/'
    data_inicio = "2024-08-01"
    data_fim = "2024-08-30"
    codigos = ['40','46','62']
    lista_proposicoes = []
    for codigo in codigos:
        url = f'{camara_base_url}/proposicoes?dataInicio={data_inicio}&dataFim={data_fim}&codTema={codigo}&ordem=ASC&ordenarPor=id'
        
        # Execucao da primeira pagina de resultados.
        response = requests.get(url, timeout=30)
        if not response.ok: # != 200:
            raise Exception('Nao foi possivel recuperar os dados')
        df_proposicoes = pd.DataFrame().from_dict(json.loads(response.text)['dados']).head(10)
        print('Total de proposicoes:', df_proposicoes.shape[0])
        lista_proposicoes.append(df_proposicoes)
    
    df_proposicoes = pd.concat(lista_proposicoes)
    
    try:
        df_proposicoes.to_parquet('../data/proposicoes_deputados.parquet')
    except Exception as e:
        print('Não foi possível salvar o arquivo em parquet:', e)
    return df_proposicoes


# 5B - Utilize a sumarização por chunks para resumir as proposições tramitadas no período de referência. Avalie a resposta e salve-a em data/sumarizacao_proposicoes.json
######## ==============================================================
## Questão 5B - Utilize a sumarização por chunks para resumir as proposições tramitadas no período de referência. Avalie a resposta e salve-a em data/sumarizacao_proposicoes.json
##  
##
##
def summarizer_chunk(df_proposicoes):
    class ChunkSummary():
        def __init__(self, model_name, apikey, text, window_size, overlap_size):
            self.text = text
            if isinstance(self.text, str):
                self.text = [self.text]
            self.window_size = window_size
            self.overlap_size = overlap_size
            # Aplicacao dos chunks
            self.chunks = self.__text_to_chunks()
            self.model = self.__create_model(apikey, model_name)


        def __create_model(self, apikey, model_name):
            genai.configure(api_key=apikey)
            self.prompt_base = """
            You are an legislative assistant from the chamber of deputies.
            You will receive the #ementas# from real propositions in the format:
            Ementa: <ementa>
            
            You must create a summary of the #ementas#, pointing out the most
            relevant information, about the propositions in this period. 
            The summary output must be written as a plain JSON with field 'summary' and 'theme':
            {[
                {
                    "summary": "The summary of the propositions",
                    "theme": "The central theme of the propositions"
                }
            ]}
            
            """
            safety_settings={
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            }
            generation_config = {
                'temperature': 0.2,
                'top_p': 0.8,
                'top_k': 20,
                'max_output_tokens': 1000
            }
            return genai.GenerativeModel(
                model_name,
                system_instruction=self.prompt_base,
                generation_config = generation_config,
                safety_settings=safety_settings
            )


        
        def __text_to_chunks(self):       
            n = self.window_size  # Tamanho de cada chunk
            m = self.overlap_size  # overlap entre chunks
            return [self.text[i:i+n] for i in range(0, len(self.text), n-m)]


        def __create_chunk_prompt(self, chunk):
            propsition_lines = '\n'.join(chunk)
            prompt = f"""
            #ementas#
            {propsition_lines}
            ######
            Summarize it.
            """
            return prompt
            
        
        def __summarize_chunks(self):
            # Loop over chunks
            chunk_summaries = []
            for i, chunk in enumerate(self.chunks):
                print(f'Summarizing chunk {i+1} from {len(self.chunks)}')
                # Create prompt
                prompt = self.__create_chunk_prompt(chunk)
                response = self.model.generate_content(prompt)
                # Apendar resposta do chunk
                chunk_summaries.append(response.text)
                
                # if i == 4: break

            return chunk_summaries


        def summarize(self):
            print('Summarizing text')
            # Chamar o sumario dos chunks
            self.chunk_summaries = self.__summarize_chunks()
            # Prompt final
            summaries = '- ' + '\n- '.join(self.chunk_summaries)
            prompt = f"""
            You are an editor working on the chamber of deputies. You must summarize
            the propositions in this period. The partitioned summaries are listed below:
            {summaries}
            ######
            The summary must describe the details in the propositions, like themes, and details
            on what groups are affected.
            Write a final summary based on the partitioned summaries in JSON format with
            the field 'summary' and 'theme':
            {{[
                {{
                    "summary": "The summary of the propositions",
                    "theme": "The central theme of the propositions"
                }}
            ]}}
            """
            print('Final summarization')
            response = self.model.generate_content(prompt)
            
            return response.text


    # episode_season = 5
    # episode_id = 92
    # X = (data[(data.episode_season == episode_season) &
    #           (data.episode_id == episode_id)].sort_values('number')
    # )

    X = df_proposicoes[['ementa']]
    X['line'] = ('Ementa: ' + df_proposicoes['ementa'].fillna(''))

    summarizer = ChunkSummary(
        model_name = "gemini-1.5-flash",
        apikey = os.environ["GEMINI_KEY"],
        text = X['line'].tolist(),
        window_size = 10,
        overlap_size = 0
    )
    
    propsition_summary = summarizer.summarize()
    propsition_summary = propsition_summary.replace("```json\n",'').replace("\n```",'')
    propsition_summary = json.loads(propsition_summary)
    propsition_summary = propsition_summary[0]
    with open('../data/sumarizacao_proposicoes.json', 'w') as f:
        json.dump(propsition_summary, f)
    
    return summarizer.chunks, summarizer.chunk_summaries, propsition_summary