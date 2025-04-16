# import json
# import chromadb
# import numpy as np
# from sentence_transformers import SentenceTransformer

# class Preprocessor:
#     def __init__(self, json_file, db_path="../../Database/ChromaDB"):
#         self.json_file = json_file
#         self.db_path = db_path
#         self.data = self.load_json()
#         self.embedding_model = SentenceTransformer("sentence-transformers/multi-qa-mpnet-base-dot-v1")
#         self.chroma_client = chromadb.PersistentClient(path=self.db_path)
#         self.collection = self.chroma_client.get_or_create_collection(name="rag_collection")
    
#     def load_json(self):
#         with open(self.json_file, "r", encoding="utf-8") as file:
#             data = json.load(file)
#         return data
    
#     def clean_metadata(self, item):
#         """ Menghapus nilai None dari metadata """
#         return {k: (v if v is not None else "") for k, v in item.items()}

#     def create_embeddings_and_store(self):
#         combined_texts = [
#             f"{item.get('title', '')} {item.get('seller_name', '')} {item.get('final_price', '')} {item.get('description', '')}"
#             for item in self.data
#         ]
#         embeddings = self.embedding_model.encode(combined_texts).tolist()
#         for idx, (embedding, item) in enumerate(zip(embeddings, self.data)):
#             cleaned_metadata = self.clean_metadata(item)
#             self.collection.add(ids=[str(idx)], embeddings=[embedding], metadatas=[cleaned_metadata])
#         print("Data telah diproses dan disimpan di ChromaDB.")

# if __name__ == "__main__":
#     preprocessor = Preprocessor('data_barang.json')
#     preprocessor.create_embeddings_and_store()


import json
import chromadb
import numpy as np
import re
from sentence_transformers import SentenceTransformer

class Preprocessor:
    def __init__(self, json_file, db_path="../../Database/ChromaDB"):
        self.json_file = json_file
        self.db_path = db_path
        self.data = self.load_json()
        self.embedding_model = SentenceTransformer("sentence-transformers/multi-qa-mpnet-base-dot-v1")
        self.chroma_client = chromadb.PersistentClient(path=self.db_path)
        self.collection = self.chroma_client.get_or_create_collection(name="rag_collection")

    def load_json(self):
        with open(self.json_file, "r", encoding="utf-8") as file:
            data = json.load(file)
        return data

    def clean_text(self, text):
        """Membersihkan teks jika bertipe string, biarkan nilai non-string tetap."""
        if isinstance(text, str):
            text = text.replace("\r", " ").replace("\n", " ").replace("\t", " ")
            text = re.sub(r'\s+', ' ', text).strip()
            return text
        return text  # biarkan angka, float, atau tipe lain tetap

    def clean_metadata(self, item):
        """ Membersihkan teks untuk string dan melewati nilai non-string dengan aman """
        return {k: self.clean_text(v) if v is not None else "" for k, v in item.items()}

    def create_embeddings_and_store(self):
        combined_texts = [
            self.clean_text(f"{item.get('title', '')} {item.get('description', '')} {item.get('categories', '')} {item.get('breadcrumbs', '')}")
            for item in self.data
        ]
        embeddings = self.embedding_model.encode(combined_texts).tolist()
        for idx, (embedding, item) in enumerate(zip(embeddings, self.data)):
            cleaned_metadata = self.clean_metadata(item)
            self.collection.add(ids=[str(idx)], embeddings=[embedding], metadatas=[cleaned_metadata])
        print("Data telah diproses, dibersihkan, dan disimpan di ChromaDB.")

if __name__ == "__main__":
    preprocessor = Preprocessor('data_barang.json')
    preprocessor.create_embeddings_and_store()
