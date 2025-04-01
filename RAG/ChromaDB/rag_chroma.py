# import chromadb
# import numpy as np
# from sentence_transformers import SentenceTransformer

# class RAG:
#     def __init__(self, db_path="../../Database/ChromaDB"):
#         self.db_path = db_path
#         self.embedding_model = SentenceTransformer("sentence-transformers/multi-qa-mpnet-base-dot-v1")
#         self.chroma_client = chromadb.PersistentClient(path=self.db_path)
#         self.collection = self.chroma_client.get_or_create_collection(name="rag_collection")
    
#     def retrieve_documents(self, query, top_k=1):
#         query_embedding = self.embedding_model.encode([query]).tolist()[0]
#         results = self.collection.query(query_embeddings=[query_embedding], n_results=top_k)
        
#         retrieved_docs = []
#         for doc_id, metadata in zip(results["ids"], results["metadatas"]):
#             retrieved_docs.append({"id": doc_id, "metadata": metadata})
        
#         return retrieved_docs

# if __name__ == "__main__":
#     rag = RAG()
#     query = "Laptop gaming murah"
#     results = rag.retrieve_documents(query)
#     for result in results:
#         print(result)


import chromadb
import numpy as np
import json
from sentence_transformers import SentenceTransformer
from LLM.groq_runtime import GroqRunTime
from sklearn.metrics.pairwise import cosine_similarity

class RagChroma:
    def __init__(self, db_path="/Users/cal/Documents/Coding/Python/Indonesia’s_Product_Recomendation_Chatbot_using_Product_Description/Database/ChromaDB"):
        self.db_path = db_path
        self.embedding_model = SentenceTransformer("sentence-transformers/multi-qa-mpnet-base-dot-v1")
        self.chroma_client = chromadb.PersistentClient(path=self.db_path)
        self.collection = self.chroma_client.get_or_create_collection(name="rag_collection")
    
    def retrieve_documents(self, query, top_k=5):
        query_embedding = self.embedding_model.encode([query]).tolist()[0]
        results = self.collection.query(query_embeddings=[query_embedding], n_results=top_k)
        
        retrieved_docs = []
        for metadata_list in results["metadatas"]:  
            if metadata_list:  
                retrieved_docs.extend(metadata_list)
        
        return retrieved_docs
    
    def sort_resources(self, query, resources):
        combined_texts = [
            f"{r.get('title', '')} {r.get('seller_name', '')} {r.get('final_price', '')} {r.get('description', '')}"
            for r in resources
        ]
        resource_embeddings = self.embedding_model.encode(combined_texts)
        query_embedding = self.embedding_model.encode([query])

        similarities = cosine_similarity(query_embedding, resource_embeddings)[0]
        most_similar_idx = np.argmax(similarities)

        return resources[most_similar_idx]
    
    def get_summary(self, query):
        groq_run = GroqRunTime()
        system_prompt = "Anda adalah asisten pencarian barang berdasarkan deskripsi berbahasa Indonesia. Tolong ubah atau ringkas query pengguna agar lebih jelas untuk pencarian produk. dan hanya respon dengan ringkasan query pengguna agar lebih jelas untuk pencarian produk."
        response = groq_run.generate_response(system_prompt, query)
        return response.choices[0].message.content
    
    def rag_search(self, query):
        refined_query = self.get_summary(query)

        retrieved_resources = self.retrieve_documents(refined_query)

        best_resource = self.sort_resources(refined_query, retrieved_resources)

        groq_run = GroqRunTime()
        system_prompt = f"Anda adalah asisten pencarian barang berdasarkan deskripsi berbahasa Indonesia. Berikut adalah detail barang yang paling mirip:\n{best_resource}"
        response = groq_run.generate_response(system_prompt, query)

        return {
            "best_match": best_resource,
            "refined_query": refined_query,
            "llm_response": response.choices[0].message.content
        }


# if __name__ == "__main__":
#     rag = RAG()
#     query = "Laptop gaming murah"
#     results = rag.retrieve_documents(query)
#     for result in results:
#         print(result)

