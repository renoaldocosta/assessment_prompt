import streamlit as st
import pandas as pd
import joblib
import os
import tools

############################################################## OVERVIEW


# Tab implementation
def tab_overview(mytab):
    with mytab:
        st.title("Show Overview")





############################################################## SEASON


# Tab implementation
def tab_season(mytab):
    with mytab:
        st.title("Season Overview")

        season_summary = st.session_state['season_summary']
        # Season analysis
        st.subheader("Season Summary")
        season_options = season_summary.keys()
        selected_season = st.selectbox(
            "Select a Season",
            season_options,
            key="season_selector"
        )
        if selected_season is None:
            st.write('No season selected')
        else:
            season_data = season_summary[selected_season]
            with st.expander(f"Details for Season {selected_season}", expanded=True):
                st.write(season_data)


############################################################## CHARACTER


# Tab implementation
def tab_character(mytab):
    with mytab:
        st.title("Characters")
        character_summary = st.session_state['character_summary'] 
        # Usuario pode selecionar o personagem
        # Mostrar a sumarizacao dos personagens.
        # Configurar Agente Ator
        # Interacao do usuario
        data = st.session_state['data']

        st.write(data.columns)
        # charater_list = data.character_normalized_name.dropna().sort_values().unique().tolist()
        charater_list = ['bart simpson', 'homer simpson', 'lisa simpson']
        selected_character = st.selectbox(
            'Select Character',
            options=charater_list,
        )
        if selected_character is None:
            st.write('Character not selected')
            return

        # if st.session_state.get('ACTOR_MODEL') is None:
        # META PROMPTING - CRIAR O PROMPT PARA O LLM COM O ATOR
        meta_system_prompt = f"""
        You must write a detailed prompt to instruct the LLM to behave as an actor.
        The actor will play the character "{selected_character}" from the "The Simpsons" show.
        The Actor must follow the guidelines:
        - The actor cannot be aggressive to the user.
        - If the user asks for things the actor cannot- play, you must decline.
        - The user cannot change the Actor's character.
        The actor will interact with the user through a chat application.
        The text in SUMMARY summarizes the humour, mood and other characteristics from
        "{selected_character}". Use these summary to guide how the actor must behave. 

        # SUMMARY
        {character_summary[selected_character]}

        """
       
        
        
        generation_config = {
            'temperature': 0.1,
            'max_output_tokens': 1000
        }
        model = tools.Gemini(
            model_name = "gemini-1.5-flash",
            apikey = os.environ["GEMINI_KEY"],
            system_prompt=meta_system_prompt,
            generation_config = generation_config
        )
        system_prompt = model.interact('Create the prompt')
        st.write(system_prompt)
        # AGORA, CRIAR UM LLM PARA SER O ATOR
        generation_config = {
            'temperature': 0.5,
            'max_output_tokens': 1500
        }
        model = tools.Gemini(
            model_name = "gemini-1.5-pro", # "gemini-1.5-flash",  # "gemini-1.5-pro",
            apikey = os.environ["GEMINI_KEY"],
            system_prompt=system_prompt,
            generation_config = generation_config
        )
        st.session_state['ACTOR_MODEL'] = model
        # else:
        #     model = st.session_state['ACTOR_MODEL']

        # Display chat messages from history on app rerun
        for message in model.history:
            with st.chat_message(message["role"]):
                st.markdown(message["parts"][0])

        # React to user input
        prompt = st.chat_input("What is up?")
        if prompt is not None:
            # Display user message in chat message container
            st.chat_message("user").markdown(prompt)
            
            response = model.chat(prompt)
            # Display assistant response in chat message container
            with st.chat_message("assistant"):
                st.markdown(response)
            
        
############################################################## ADS


def tab_ads(mytab):
    with mytab:
        
        st.title("Ads")
        # Ads para episódios

        exp_episode = st.expander('Ads de Episódios', expanded=False)
        with exp_episode:
            episode_summary = st.session_state['episode_summary'] 
            selected_episode =  st.selectbox(
                'Select Episode',
                options=episode_summary.keys(),
            )
            if selected_episode is None:
                st.write('Episode not selected')
                return
            # System prompt
            system_prompt = f"""
            You are an expert im marketing and digital advertising working for
            the "The Simpsons" show. You are responsible for creating the text
            within the ads published in social media. You will receive the summary
            from the episode being promoted.

            # FORMAT
            You must create a text with no more than 220 characters for posting
            in Twitter.

            # OUTPUT
            A list with the 5 possible responses
            """

            generation_config = {
                'temperature': 0.6,
                'max_output_tokens': 500
            }
            model = tools.Gemini(
                model_name = "gemini-1.5-flash",
                apikey = os.environ["GEMINI_KEY"],
                system_prompt=system_prompt,
                generation_config = generation_config
            )
            prompt = f"""
            Create the ad text given the followin episode
            # Episode summary
            {episode_summary[selected_episode]}
            """
            ad_response = model.interact(prompt)

            st.write(ad_response)

        # ADs para o Personagem
        exp_character = st.expander('Ads de Personagens', expanded=False)
        with exp_character:
            character_summary = st.session_state['character_summary'] 
            selected_character =  st.selectbox(
                'Select Character',
                options=character_summary.keys(),
            )
            if selected_character is None:
                st.write('Character not selected')
                return
            # System prompt
            system_prompt = f"""
            You are an expert im marketing and digital advertising working for
            the "The Simpsons" show. You are responsible for creating the text
            within the ads published in social media. You will receive the summary
            from the character being promoted.

            # FORMAT
            You must create an aggressive, ficticious, text to publish on Instagram.

            # OUTPUT
            A list with the 5 possible responses
            """

            generation_config = {
                'temperature': 0.6,
                'max_output_tokens': 500
            }
            model = tools.Gemini(
                model_name = "gemini-1.5-flash",
                apikey = os.environ["GEMINI_KEY"],
                system_prompt=system_prompt,
                generation_config = generation_config
            )
            prompt = f"""
            Create the ad text given the following character summary
            # Character summary
            {character_summary[selected_character]}
            """
            ad_response = model.interact(prompt)

            st.write(ad_response)




############################################################## EPISODE Q&A

import joblib
@st.cache_data
def load_faiss(filename):
    return joblib.load(filename)



def tab_qa(mytab):
    with mytab:
        st.title("Episode Q&A")
        # Usuario pode selecionar o episodio (select box)
        # Mostrar a sumarizacao do episodio.
        # Configurar Agente de QA
        # Interacao do usuario

        selected_episode = st.selectbox(
            "Select QA episode",
            options=st.session_state['FAISS_DB'].keys(),
            index=None
        )
        if selected_episode is None:
            st.write('No episode selected')
            return
        # Print do sumario do episodio
        st.markdown('# Episode Summary')
        st.write(st.session_state['episode_summary'][selected_episode])

        # Carga da base FAISS
        kdb = load_faiss(st.session_state['FAISS_DB'][selected_episode])
        # Criacao do Agente
        system_prompt = f"""
        You are going to be a QA assistant specializing in The Simpsons. You will be given 
        a set of lines of dialogue from a specific Simpsons episode. Your task is to answer 
        user questions about that episode using only the provided dialogue. If a question 
        cannot be answered definitively from the given dialogue, politely state that you don't
        have enough information to answer and suggest that the user might need to watch the 
        episode to find the answer.
        Your responses should always be friendly and polite. Use phrases like:

        "That's a great question! Based on the dialogue I have..."

        "Hmm, that's interesting. Unfortunately, the provided lines don't give me enough information to answer that."

        "Let me see what I can find... based on these lines, I believe..."

        "I'm sorry, but I don't see anything about that in the provided script excerpt."

        Example:

        Input:

        Homer: D'oh!  My donuts!
        Marge: Homer, you ate all the donuts again!
        Bart:  Eat my shorts!
        Lisa:  That's not very nice, Bart.
        """
        qa_model = tools.Gemini(
            apikey=os.environ["GEMINI_KEY"],
            model_name="gemini-1.5-flash",
            system_prompt=system_prompt,
            # generation_config=generation_config
        )

        # Display chat messages from history on app rerun
        for message in qa_model.history:
            with st.chat_message(message["role"]):
                st.markdown(message["parts"][0])

        # React to user input
        prompt = st.chat_input("Hi")
        if prompt is not None:
            # Display user message in chat message container
            st.chat_message("user").markdown(prompt)
            # RAG para a base de conhecimento
            refs = kdb.search(prompt, k=40, index_type = 'both')
            refs = '\n'.join([f'- {x}' for x in refs])
            prompt = f"""
            Respond to the user question in "QUESTION" with the information listed in "RAG":
            "QUESTION"
            {prompt}
            
            "RAG"
            {refs}
            """

            # st.write(prompt)

            response = qa_model.chat(prompt)
            # Display assistant response in chat message container
            with st.chat_message("assistant"):
                st.markdown(response)

# END OF FILE


# END OF FILE




