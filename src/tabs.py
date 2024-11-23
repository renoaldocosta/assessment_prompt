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


def tab_episode_qa(mytab):
    with mytab:
        

        st.title("Episode Q&A")


        # Usuario pode selecionar o episodio
        # Mostrar a sumarizacao do episodio.
        # Configurar Agente de QA
        # Interacao do usuario

        # import streamlit as st
        # from langchain import LLMChain, OpenAI
        # from langchain.agents import AgentExecutor, Tool, ConversationalAgent
        # from langchain.memory import ConversationBufferMemory
        # from langchain_community.llms import OpenAI
        # # from langchain.utilities import OpenWeatherMapAPIWrapper
        # # from langchain.utilities import GoogleSerperAPIWrapper
        # from langchain_community.chat_models import ChatOpenAI
        # from langchain.memory.chat_message_histories import StreamlitChatMessageHistory

        # import tools


        # st.title("Character Overview")


        # AI = OpenAI(temperature=0.7)

        # # the simpson tool
        # if st.session_state.get('FAISS_DB') is None:
        #     st.warning('Could not find objecto FAISS DB')
        #     return

        # # Criacao das tools que vao auxiliar no prompt do agente
        # # Colocar os tools de busca em cada episodio.
        # simpsons_episode_tools = []
        # for k,v in st.session_state['FAISS_DB'].items():
        #     simpsons_episode_tools.append(
        #         tools.KDBFaissTool(
        #             v,
        #             f'episode {k} assistant',
        #             f"""AI Assistant for understanding, questioning and explaining the
        #             story in episode {k} of the 'The Simpsons show'.
        #             """
        #         )
        #     )


        # # # search tool
        # # search = GoogleSerperAPIWrapper(serper_api_key=SERPER_API_KEY)
        # # # weather tool
        # # weather = OpenWeatherMapAPIWrapper(
        # #     openweathermap_api_key=OPENWEATHERMAP_API_KEY
        # # )

        # tools = []
        # tools.extend(simpsons_episode_tools)
        # # tools = [
        # #     Tool(
        # #         name="Search",
        # #         func=search.run,
        # #         description="Useful for when you need to get current, up to date answers.",
        # #     ),
        # #     Tool(
        # #         name="Weather",
        # #         func=weather.run,
        # #         description="Useful for when you need to get the current weather in a location.",
        # #     ),
        # # ]


        # prefix = """
        # You are an actor who will play the character "{selected_character}" from "The Simpsons" show
        # assistant for the "The Simpsons" show. You help users understand what happened in the series and
        # details of specific episodes. You will 
        # You are a friendly, modern day planner. You help users find activities in a given city based on their preferences and the weather.
        #             You have access to to two tools:"""
        # suffix = """Begin!"
        # Chat History:
        # {chat_history}
        # Latest Question: {input}
        # {agent_scratchpad}"""

        # prompt = ConversationalAgent.create_prompt(
        #     tools,
        #     prefix=prefix,
        #     suffix=suffix,
        #     input_variables=["input", "chat_history", "agent_scratchpad"],
        # )

        # msgs = StreamlitChatMessageHistory()

        # if "memory" not in st.session_state:
        #     st.session_state.memory = ConversationBufferMemory(
        #         messages=msgs, memory_key="chat_history", return_messages=True
        #     )

        # memory = st.session_state.memory

        # llm_chain = LLMChain(
        #     llm=ChatOpenAI(temperature=0.8, model_name="gpt-4"),
        #     prompt=prompt,
        #     verbose=True,
        # )

        # agent = ConversationalAgent(
        #     llm_chain=llm_chain,
        #     tools=tools,
        #     verbose=True,
        #     memory=memory,
        #     max_iterations=3,
        # )


        # agent_chain = AgentExecutor.from_agent_and_tools(
        #     agent=agent, tools=tools, verbose=True, memory=memory
        # )


        # query = st.text_input(
        #     "What are you in the mood for?",
        #     placeholder="I can help!",
        # )

        # if query:
        #     with st.spinner("Thinking...."):
        #         res = agent_chain.run(query)
        #         st.info(res, icon="🤖")


        # with st.expander("My thinking"):
        #     st.write(st.session_state.memory.chat_memory.messages)





# END OF FILE


# END OF FILE




