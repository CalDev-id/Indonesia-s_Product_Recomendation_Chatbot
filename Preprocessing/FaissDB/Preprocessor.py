
# import json
# import faiss
# import pandas as pd
# from sentence_transformers import SentenceTransformer

# class Preprocessor:
#     def __init__(self):
#         # Konversi CSV ke JSON
#         csv_file = '../../rawdata/Tokopedia_Products.csv'  
#         json_file = 'data_barang.json'
#         self.convert_csv_to_json(csv_file, json_file)

#         # Load JSON data
#         with open(json_file, 'r') as f:
#             self.data = json.load(f)

#         # Inisialisasi model embedding
#         self.embedding_model = SentenceTransformer('sentence-transformers/multi-qa-mpnet-base-dot-v1')

#     def convert_csv_to_json(self, csv_path, json_path):
#         """Membaca CSV dan menyimpan sebagai JSON"""
#         df = pd.read_csv(csv_path)
#         df.to_json(json_path, orient='records', indent=4)

#     def create_embeddings(self):
#         # Membuat representasi teks gabungan dari JSON
#         combined_texts = [
#             f"{item['title']} {item['description']} {item['seller_name']}"
#             for item in self.data
#         ]
        
#         # Generate embeddings
#         embeddings = self.embedding_model.encode(combined_texts)

#         # Buat FAISS index
#         dimension = embeddings.shape[1]
#         index = faiss.IndexFlatL2(dimension)
#         index.add(embeddings)

#         # Simpan FAISS index
#         faiss.write_index(index, '../../Database/Faiss/faiss_index.index')

#         # Simpan metadata JSON
#         with open('../../Database/Faiss/metadata.json', 'w') as f:
#             json.dump(self.data, f, indent=4)

# if __name__ == "__main__":
#     preprocessor = Preprocessor()
#     preprocessor.create_embeddings()
#     print("CSV dikonversi ke JSON, embedding dibuat, dan FAISS index disimpan.")


import json
import faiss
import pandas as pd
import re
from sentence_transformers import SentenceTransformer

class Preprocessor:
    def __init__(self):
        # Konversi CSV ke JSON
        csv_file = '../../rawdata/Tokopedia_Products.csv'  
        json_file = 'data_barang.json'
        self.convert_csv_to_json(csv_file, json_file)

        # Load JSON data
        with open(json_file, 'r', encoding='utf-8') as f:
            self.data = json.load(f)

        # Inisialisasi model embedding
        self.embedding_model = SentenceTransformer('sentence-transformers/multi-qa-mpnet-base-dot-v1')

    def convert_csv_to_json(self, csv_path, json_path):
        """Membaca CSV dan menyimpan sebagai JSON"""
        df = pd.read_csv(csv_path)
        df.to_json(json_path, orient='records', indent=4)

    def clean_text(self, text):
        if isinstance(text, str):
            text = text.replace('\r', ' ').replace('\n', ' ').replace('\t', ' ')
            text = re.sub(r'\s+', ' ', text)
            return text.strip()
        return text  # biarkan nilai non-str tetap seperti semula

    def clean_data(self):
        for item in self.data:
            for key in item:
                item[key] = self.clean_text(item[key])


    def create_embeddings(self):
        # Bersihkan data terlebih dahulu
        self.clean_data()

        # Buat representasi teks gabungan
        combined_texts = [
            f"{item.get('title', '')} {item.get('description', '')} {item.get('categories', '')} {item.get('breadcrumbs', '')}"
            for item in self.data
        ]

        # Generate embeddings
        embeddings = self.embedding_model.encode(combined_texts)

        # Buat FAISS index
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings)

        # Simpan FAISS index
        faiss.write_index(index, '../../Database/Faiss/faiss_index.index')

        # Simpan metadata yang sudah dibersihkan
        with open('../../Database/Faiss/metadata.json', 'w', encoding='utf-8') as f:
            json.dump(self.data, f, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    preprocessor = Preprocessor()
    preprocessor.create_embeddings()
    print("CSV dikonversi ke JSON, data dibersihkan, embedding dibuat, dan FAISS index disimpan.")
