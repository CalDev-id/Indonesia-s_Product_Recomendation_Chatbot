import numpy as np
import faiss
import json
from sentence_transformers import SentenceTransformer
from LLM.groq_runtime import GroqRunTime

class RagFaiss:
    def __init__(self):
        # Load Faiss index and metadata

        # database only description
        # self.index = faiss.read_index('Database/faiss_index.index')
        # with open('Database/metadata.json', 'r') as f:
        #     self.metadata = json.load(f)

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
    
    def rag_search(self, query):
        # Step 1: Retrieve top-k resources from Faiss
        retrieved_resources = self.search_faiss(query, k=50)

        # Step 2: Sort resources based on similarity with the query
        best_resource = self.sort_resources(query, retrieved_resources)

        # Step 3: Reasoning using LLM (Groq LLaMA3-70B-8192)
        groq_run = GroqRunTime()
        system_prompt = f"Anda adalah asisten pencarian barang berdasarkan deskripsi berbahasa indonesia. Berikut adalah detail barang yang paling mirip:\n{best_resource['title'], best_resource['description'], best_resource['seller_name']}"
        response = groq_run.generate_response(system_prompt, query)

        return {
            "best_match": best_resource,
            "llm_response": response.choices[0].message.content
        }