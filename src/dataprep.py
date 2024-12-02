
import time
import pandas as pd
import os
import joblib
from dotenv import load_dotenv
load_dotenv('../.env')

import faiss
from sentence_transformers import SentenceTransformer
import numpy as np
from summarizer import ChunkSummary
from kdb_faiss import KDBFaiss

EPISODE_IDS = os.environ.get('EPISODE_IDS')
EPISODE_IDS = [int(x) for x in EPISODE_IDS.split(',')]

CHARACTERS = os.environ.get('CHARACTERS')
CHARACTERS = CHARACTERS.split(',')

EXECUTE_SUMMARIZATION_EPISODE = os.environ.get('EXECUTE_SUMMARIZATION_EPISODE', False)
EXECUTE_SUMMARIZATION_SEASON = os.environ.get('EXECUTE_SUMMARIZATION_SEASON', False)
EXECUTE_SUMMARIZATION_CHARACTER = os.environ.get('EXECUTE_SUMMARIZATION_CHARACTER', False)
EXECUTE_EPISODES_MNLI = os.environ.get('EXECUTE_EPISODES_MNLI', False)
EXECUTE_FAISS_KDB = os.environ.get('EXECUTE_FAISS_KDB', True)


#########################################################
## CARGA DOS DADOS:
##

DBFILE = os.environ.get('DBFILE')
data = pd.read_parquet(DBFILE)
data = data.sort_values('number')
print('DATA READ', data.shape)


#########################################################
## TAREFA 1:
## Sumarização de episódios:
## Utilizar técnicas de prompt engineering para sumarizar um episódio do programa.


if EXECUTE_SUMMARIZATION_EPISODE:

    episode_summaries = {}

    for episode_id in EPISODE_IDS:
        X = data[data.episode_id == episode_id].copy()
        X = ("Episode " + X['episode_id'].astype(str) + ' | ' + 
                        X['location_normalized_name'].fillna('') + ', ' + 
                        X['character_normalized_name'].fillna('') + ' said: ' + 
                        X['normalized_text'].fillna('')
        )

        print(X.shape)

        system_prompt = f"""
        You are an editor assistant from the "The Simpsons" show.
        You will receive chunk of subtitles from real episodes in the format:
        <episode number> | <location>, <character> said: <character line>

        You must create a summary of the episode, pointing out the most
        relevant information and key players in the story. Bare in mind
        that the summary must describe how the episode started, which key
        points are relevant along the story and its gran finale.
        """

        generation_config = {
            'temperature': 0.2,
            # 'top_p': 0.8,
            # 'top_k': 20,
            'max_output_tokens': 200
        }

        summarizer = ChunkSummary(    
            model_name = "gemini-1.5-flash",
            apikey = os.environ["GEMINI_KEY"],
            text = X.tolist(),
            window_size = 100,
            overlap_size = 10,
            system_prompt=system_prompt,
        )
        episode_summaries[episode_id] = summarizer.summarize()
        time.sleep(10)
        
    # Exportacao
    joblib.dump(episode_summaries, '../data/summaries/summary_episodes.joblib', compress=9)
else:
    episode_summaries = joblib.load('../data/summaries/summary_episodes.joblib')


#########################################################
## TAREFA 2
## Sumarização da temporada:
## Quebrar a temporada em chunks de episódios para sumarização dividida.

if EXECUTE_SUMMARIZATION_SEASON:
    # Recuperar somente temporadas cujos episodios ja foram resumidos
    # Batch-prompting com principio de divisao de taregas.
    # Sumarizar pelos sumarios dos episodios
    df = pd.DataFrame().from_dict(episode_summaries, orient='index')
    df = df.reset_index()
    df.columns = ['episode_id','episode_summary']
    df = df.merge(data, on='episode_id')
    df = df[['episode_id','episode_season','episode_summary']].drop_duplicates()
    df = df.sort_values(['episode_season','episode_id'])
    # Iteracao sobre temporadas
    season_summaries = {}
    for episode_season in df.episode_season.unique():
        X = df[df.episode_season == episode_season].copy()
        X = ("Episode " + X['episode_id'].astype(str) +
             ' | ' + 
             X['episode_summary']
        )

        print(X.shape)

        # SELF-ASK PROMPT
        system_prompt = f"""
        You are an editor assistant from the "The Simpsons" show.
        You will receive the summaries of the episodes from the show in the format:
        Episode <episode number> | <episode summary>

        ## INSTRUCTION
        You must create a summary of the season based on the summary of each episode,
        pointing out the most relevant information and key players in the story such as:
        - What relevant happened to the main characters?
        - Which stories are relevant to highlight for the audience?
        - Which jokes should be mentioned in order to promote the season?
        """

        generation_config = {
            'temperature': 0.2,
            # 'top_p': 0.8,
            # 'top_k': 20,
            'max_output_tokens': 200
        }

        summarizer = ChunkSummary(    
            model_name = "gemini-1.5-flash",
            apikey = os.environ["GEMINI_KEY"],
            text = X.tolist(),
            window_size = 2,
            overlap_size = 1,
            system_prompt=system_prompt,
        )

        season_summaries[int(episode_season)] = summarizer.summarize()
        time.sleep(30)

    # Exportacao
    joblib.dump(season_summaries, '../data/summaries/summary_seasons.joblib', compress=9)
else:
    season_summaries = joblib.load('../data/summaries/summary_seasons.joblib')


#########################################################
## TAREFA 3
## Sumarizador de personas:
## Utilizar técnicas de chunks para consolidar as características do personagem segundo suas falas.
## Tecnica Self-Ask com exemplos de resposta
##

if EXECUTE_SUMMARIZATION_CHARACTER:
    # Recuperar somente temporadas cujos episodios ja foram resumidos
    X = (data[
            (data.character_normalized_name.isin(CHARACTERS)) &
            (data.episode_id.isin(EPISODE_IDS))]
            .sort_values(['episode_id','number'])
    )
    # Iteracao em personagens
    character_summaries = {}
    for character_name in X.character_normalized_name.unique():
        Xc = X[X.character_normalized_name == character_name].copy()
        Xc = (X['location_normalized_name'].fillna('') + '|' + X['normalized_text'].fillna(''))
        
        system_prompt = f"""
        You are an editor assistant from the "The Simpsons" Show. 
        You will receive lines from '{character_name}', extracted directly from the episodes.
        With these LINES, you must write a SUMMARY for '{character_name}' that covers,
        among others:
            - What relevant happened to '{character_name}'?
            - Which sentiments, or moods, we can associate to '{character_name}'?
            - How did '{character_name}' reacted a different situations?
        
        # LINES
        <location name>|<character line>
        """

        generation_config = {
            'temperature': 0.2,
            # 'top_p': 0.8,
            # 'top_k': 20,
            'max_output_tokens': 200
        }
        summarizer = ChunkSummary(    
            model_name = "gemini-1.5-flash",
            apikey = os.environ["GEMINI_KEY"],
            text = Xc.tolist(),
            window_size = 400,
            overlap_size = 10,
            system_prompt=system_prompt,
        )
        character_summaries[character_name] = summarizer.summarize()
        time.sleep(10)
        
        break
        
    # Exportacao
    joblib.dump(character_summaries, '../data/summaries/summary_characters_2.joblib', compress=9)
else:
    character_summaries = joblib.load('../data/summaries/summary_characters_2.joblib')





#########################################################
## TAREFA 4
## Classificação de sentimento das falas:
## Criar IA que classifica os episodios de acordo com labels MNLI. 





#########################################################
## TAREFA 5
## Classificador de Faixa Etária com BART-MNLI:
## Utilizar técnica LLM de NLI para estimar a faixa etária dos episódios da série.
## Comparar os termos NLI com os ratings e audiência dos episódios.

# Definir conjunto de labels para o llm classificar (MNLI)

security_labels = [
    'alcohol',
    'sexual',
    'violence',
    'offensive',
    'abusive',
    'drugs',
    'racism'
]

age_disallowed_labels = {
    'kids': security_labels,
    'teens': ['alcohol', 'sexual', 'drugs', 'racism'],
    'adults': ['racism'],
}


if EXECUTE_EPISODES_MNLI:
    print('EXECUTE_EPISODES_MNLI')

    from transformers import pipeline

    # Funcao para classificacao por NLI
    def nli_classification(sequence_to_classify):
        classifier = pipeline(
            "zero-shot-classification",
            model="facebook/bart-large-mnli",
            device='cpu'
        )

        label_dict = classifier(sequence_to_classify, security_labels)
        label_dict.pop('sequence')
        return label_dict
    

    # Iteracao sobre os episodios a serem analisados
    espisode_labels = {}
    for episode_id in EPISODE_IDS:
        print('EPISODE', episode_id)
        # Extracao das labels das linhas
        X = data[(data.episode_id == episode_id)].sort_values('number')
        X = X.dropna(subset='normalized_text').sample(10)
        X['nli_labels'] = X.normalized_text.apply(lambda x: nli_classification(x))
        # Y = pd.json_normalize(X.nli_labels)
        Y = pd.DataFrame().from_records(X.nli_labels.tolist())
        scores = []
        for L,S in zip(Y['labels'], Y['scores']) :
            scores.append({l:s for l, s in zip(L,S)})

        # {l:s for l, s in zip(Y['labels'], Y['scores'])}


        df_labels = pd.DataFrame().from_records(scores)
        df_labels.index = X.index
        X = pd.concat((X, df_labels), axis=1)
        # Recuperar colunas principais para salvar
        espisode_labels[episode_id] = X[['episode_id','number',] + security_labels]

    # Exportacao
    joblib.dump(espisode_labels, '../data/security/episodes_lines_nli-2.joblib', compress=9)
else:
    espisode_labels = joblib.load('../data/security/episodes_lines_nli-2.joblib')





#####################################################################3
# BASES VETORIAIS

if EXECUTE_FAISS_KDB:
    # Iteracao sobre os episodios
    for episode_id in EPISODE_IDS:
        X = data[data.episode_id == episode_id].copy()
        X = ("Episode " + X['episode_id'].astype(str) + ' | ' + 
                        X['location_normalized_name'].fillna('') + ', ' + 
                        X['character_normalized_name'].fillna('') + ' said: ' + 
                        X['normalized_text'].fillna('')
        )
        db = KDBFaiss(
            model_name = 'all-MiniLM-L6-v2',
            cache_folder= '../data/llms',
            device = 'cpu',
        )
        # Processar o embeddings do texto e adicionar ao indice
        db.add_text(X.tolist())

        # Exportacao dos indices do KDB
        db.export_kdb(f"../data/faiss/backup-kdb_episode_id_{episode_id}.faiss")



# END OF FILE



