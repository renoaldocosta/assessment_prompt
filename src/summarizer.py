

from tools import Gemini

class ChunkSummary():
    def __init__(self, model_name, apikey, text, window_size,
                 overlap_size, system_prompt, generation_config=None):
        self.text = text
        if isinstance(self.text, str):
            self.text = [self.text]
        self.window_size = window_size
        self.overlap_size = overlap_size
        # Aplicacao dos chunks e criacao do modelo
        self.chunks = self.__text_to_chunks()

        self.model = Gemini(
            apikey=apikey,
            model_name=model_name,
            system_prompt=system_prompt,
            generation_config=generation_config)

    
    def __text_to_chunks(self):       
        n = self.window_size  # Tamanho de cada chunk
        m = self.overlap_size  # overlap entre chunks
        return [self.text[i:i+n] for i in range(0, len(self.text), n-m)]


    def __create_chunk_prompt(self, chunk):
        episode_lines = '\n'.join(chunk)
        prompt = f"""
        Summarize the chunk text:
        ###### CHUNK
        {episode_lines}
        ######
        """
        return prompt
        
    
    def __summarize_chunks(self):
        # Loop over chunks
        chunk_summaries = []
        for i, chunk in enumerate(self.chunks):
            print(f'Summarizing chunk {i+1} from {len(self.chunks)}')
            # Create prompt
            prompt = self.__create_chunk_prompt(chunk)
            response = self.model.interact(prompt)
            # Apendar resposta do chunk
            chunk_summaries.append(response)
            
            # if i == 4: break

        return chunk_summaries


    def summarize(self):
        print('Summarizing text')
        # Chamar o sumario dos chunks
        self.chunk_summaries = self.__summarize_chunks()
        # Prompt final
        summaries = [f"- {x}\n" for x in self.chunk_summaries]
        prompt_summary = f"""
        Summarize the information in ### chunk summaries.

        ### chunk summaries
        {summaries}
        ###

        Write the output in raw text with the summary only.

        """
        print('Interacting')
        response = self.model.interact(prompt_summary)
        
        return response
        


