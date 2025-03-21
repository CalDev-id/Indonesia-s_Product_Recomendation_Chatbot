import numpy as np
import faiss
import json
from sentence_transformers import SentenceTransformer
from LLM.groq_runtime import GroqRunTime

class RagFaiss:
    def __init__(self):
        # database all
        self.index = faiss.read_index("C:/Users/haica/Documents/PAPER NLP/Indonesia-s_Product_Recomendation_Chatbot/Database/Faiss/faiss_index.index")
        with open("C:/Users/haica/Documents/PAPER NLP/Indonesia-s_Product_Recomendation_Chatbot/Database/Faiss/metadata.json", "r") as f:
            self.metadata = json.load(f)


        # Initialize Sentence Transformer model
        self.embedding_model = SentenceTransformer('sentence-transformers/multi-qa-mpnet-base-dot-v1')

    # Function to search for relevant items in Faiss
    def search_faiss(self, query, k=50):
        query_embedding = self.embedding_model.encode([query])
        D, I = self.index.search(query_embedding, k)  # Search the top-k items
        results = [self.metadata[i] for i in I[0]]
        return results

    # Function to sort and find the most similar resource using Sentence Transformer
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
        # Step 1: Perbaiki atau ringkas query menggunakan LLM
        refined_query = self.get_summary(query)

        # Step 2: Retrieve top-k resources from Faiss
        retrieved_resources = self.search_faiss(refined_query, k=50)

        # Step 3: Sort resources based on similarity with the refined query
        best_resource = self.sort_resources(refined_query, retrieved_resources)

        # Step 4: Gunakan LLM untuk memberikan jawaban yang lebih informatif berdasarkan hasil pencarian
        groq_run = GroqRunTime()
        system_prompt = f"Anda adalah asisten pencarian barang berdasarkan deskripsi berbahasa Indonesia. Berikut adalah detail barang yang paling mirip:\n{best_resource['title']}, {best_resource['description']}, {best_resource['seller_name']}"
        response = groq_run.generate_response(system_prompt, refined_query)

        return {
            "best_match": best_resource,
            "refined_query": refined_query,
            "llm_response": response.choices[0].message.content
        }