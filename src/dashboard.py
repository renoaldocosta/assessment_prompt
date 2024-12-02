

import streamlit as st
import pandas as pd
import os
import joblib
from dotenv import load_dotenv
load_dotenv('../.env')

from kdb_faiss import KDBFaiss


import tabs

# Cache data loading
@st.cache_data
def load_data(file_path):
    data = pd.read_parquet(file_path)
    data = data.sort_values('number')
    return data


# Cache data loading for season and episode summaries
@st.cache_data
def load_joblib(file_path):
    return joblib.load(file_path)


############################################################# INICIO
st.title("The Simpsons Show: Data Explorer")


############################################################# CARGA DOS DADOS BRUTOS
dbfile = os.environ.get('DBFILE')
if not dbfile:
    st.error("DBFILE environment variable is not set.")
if st.session_state.get('data',None) is None:
    try:
        data = load_data(dbfile)
        st.session_state['data'] = data
    except Exception as e:
        st.error(f"Error loading data: {e}")
data = st.session_state['data']


############################################################# CARGA DOS SUMARIOS
# Load season and episode summaries
episode_summary_file = os.environ.get("EPISODE_SUMMARY_FILE")
st.session_state['episode_summary'] = load_joblib(episode_summary_file)

character_summary_file = os.environ.get("CHARACTER_SUMMARY_FILE")
st.session_state['character_summary'] = load_joblib(character_summary_file)

season_summary_file = os.environ.get("SEASON_SUMMARY_FILE")
st.session_state['season_summary'] = load_joblib(season_summary_file)

############################################################# BASE DE DADOS VETORIAIS
st.session_state['FAISS_DB'] = {
    103: '../data/faiss/kdb_episode_id_103.faiss',
    60: '../data/faiss/kdb_episode_id_60.faiss',
    70: '../data/faiss/kdb_episode_id_70.faiss',
    81: '../data/faiss/kdb_episode_id_81.faiss',
    92: '../data/faiss/kdb_episode_id_92.faiss',
    93: '../data/faiss/kdb_episode_id_93.faiss',
}

############################################################# SIDE BAR

st.sidebar.title("About The Simpsons Show")
st.sidebar.write("Explore the episodes, characters, and seasons of The Simpsons.")
st.sidebar.metric("Total Seasons", data['episode_season'].nunique())
st.sidebar.metric("Total Episodes", data['episode_id'].nunique())
st.sidebar.metric("Total Characters", data['character_id'].nunique())
############################################################# ABAS

(overview_tab,
 season_tab,
 character_tab,
 ads_tab,
 qa_tab) = st.tabs(("Overview", "Seasons", "Characters", "Ads", "Episode Q&A"))

# tabs.tab_overview(overview_tab)

# tabs.tab_season(season_tab)

# tabs.tab_character(character_tab)

# tabs.tab_ads(ads_tab)

tabs.tab_qa(qa_tab)

