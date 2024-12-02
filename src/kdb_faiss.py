import faiss

from sentence_transformers import SentenceTransformer
import numpy as np
import joblib


class KDBFaiss(object):
    def __init__(self, model_name, cache_folder, device):
        # Salvar parametros
        self.model_name = model_name
        self.cache_folder = cache_folder
        self.device = device
        self.texts = []

        # Criar modelo de embeddings
        self.embedding_model = SentenceTransformer(
            model_name, 
            cache_folder=cache_folder, 
            device=device
        )

        # Indices FAISS
        self.index_l2 = None
        self.index_ip = None

   # Adiciona o embeddings ao indice FAISS
    def add_embeddings(self, embeddings):
        d = embeddings.shape[1]  # Dimensão dos embeddings
        if self.index_l2 is None:
            self.index_l2 = faiss.IndexFlatL2(d)  # Usando L2 (distância euclidiana) como métrica

        if self.index_ip is None:
            self.index_ip = faiss.IndexFlatIP(d)

        self.index_l2.add(embeddings)
        self.index_ip.add(embeddings)
        

    # Processar o embeddings do texto e adicionar ao indice
    def add_text(self, texts):
        if isinstance(texts, str):
            texts = [texts]

        self.texts = texts
        embeddings = self.embedding_model.encode(texts) #.astype("float16")
        faiss.normalize_L2(embeddings)
        self.add_embeddings(embeddings)


    # Exportacao dos indices do KDB
    def export_kdb(self, filename):
        # export = {
        #     'texts': self.texts,
        #     'index_l2': self.index_l2,
        #     'index_ip': self.index_ip
        # }
        joblib.dump(self, f"{filename}")


    # Exportacao dos indices do KDB
    @staticmethod
    def import_kdb(filename):
        return joblib.load(filename)
        
        # self.texts = export['texts']
        # self.index_l2 = export['index_l2']
        # self.index_ip = export['index_ip']
        

    # Busca de termos na base vetorial
    def search(self, query, k = 5, index_type = 'l2'):
        query_embedding = self.embedding_model.encode([query]) #.astype("float16")
        faiss.normalize_L2(query_embedding)
        results = []
        if index_type.lower() == 'l2' or index_type.lower() == 'both':
            distances, indices = self.index_l2.search(query_embedding, k)
            # Mostrar os resultados
            for i in range(k):
                # print(f"Texto mais próximo {i+1}: {self.texts[indices[0][i]]} (distância: {distances[0][i]})")
                results.append(self.texts[indices[0][i]])
        if index_type.lower() == 'ip' or index_type.lower() == 'both':
            distances, indices = self.index_ip.search(query_embedding, k)
            # Mostrar os resultados
            for i in range(k):
                # print(f"Texto mais próximo {i+1}: {self.texts[indices[0][i]]} (distância: {distances[0][i]})")
                results.append(self.texts[indices[0][i]])
        
                
        return np.unique(results)

# END OF FILE

