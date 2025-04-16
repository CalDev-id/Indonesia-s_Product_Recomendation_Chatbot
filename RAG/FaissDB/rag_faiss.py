import numpy as np
import faiss
import json
from sentence_transformers import SentenceTransformer
from LLM.groq_runtime import GroqRunTime

class RagFaiss:
    def __init__(self):
        # database all
        self.index = faiss.read_index("/Users/cal/Documents/Coding/Python/Indonesia’s_Product_Recomendation_Chatbot_using_Product_Description/Database/Faiss/faiss_index.index")
        with open("/Users/cal/Documents/Coding/Python/Indonesia’s_Product_Recomendation_Chatbot_using_Product_Description/Database/Faiss/metadata.json", "r") as f:
            self.metadata = json.load(f)

        self.embedding_model = SentenceTransformer('sentence-transformers/multi-qa-mpnet-base-dot-v1')

    def search_faiss(self, query, k=20):
        query_embedding = self.embedding_model.encode([query])
        D, I = self.index.search(query_embedding, k)
        results = [self.metadata[i] for i in I[0]]
        return results

    def sort_resources(self, query, resources):
        resource_embeddings = self.embedding_model.encode([r['description'] for r in resources])
        query_embedding = self.embedding_model.encode([query])

        similarities = np.dot(resource_embeddings, query_embedding.T).flatten()
        most_similar_idx = np.argmax(similarities)

        return resources[most_similar_idx]
    
    def get_summary(self, query):
        """ Menggunakan LLM untuk menyempurnakan query sebelum pencarian di Faiss """
        groq_run = GroqRunTime()
        system_prompt = "Anda adalah asisten pencarian barang berdasarkan deskripsi berbahasa Indonesia. Tolong ubah atau ringkas query pengguna agar lebih jelas untuk pencarian produk. dan hanya respon dengan ringkasan query pengguna agar lebih jelas untuk pencarian produk."
        response = groq_run.generate_response(system_prompt, query)
        return response.choices[0].message.content
    
    def rag_search(self, query):
        refined_query = self.get_summary(query)

        retrieved_resources = self.search_faiss(refined_query, k=50)

        best_resource = self.sort_resources(refined_query, retrieved_resources)

        groq_run = GroqRunTime()
        system_prompt = f"Anda adalah asisten pencarian barang berdasarkan deskripsi berbahasa Indonesia. jawab pertanyaan user berdasarkan informasi produk ini:\n judul produk :{best_resource['title']}, deskripsi:{best_resource['description']}, kategori:{best_resource['categories']}, Harga:{best_resource['final_price']}, nama penjual:{best_resource['seller_name']}, link produk:{best_resource['url']}"
        response = groq_run.generate_response(system_prompt, refined_query)

        return {
            "best_match": best_resource,
            "refined_query": refined_query,
            "llm_response": response.choices[0].message.content
        }