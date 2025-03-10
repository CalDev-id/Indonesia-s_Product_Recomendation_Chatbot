
import json
import faiss
import pandas as pd
from sentence_transformers import SentenceTransformer

class Preprocessor:
    def __init__(self):
        # Konversi CSV ke JSON
        csv_file = '../RawData/data_products_id_tiny.csv'
        json_file = 'data_barang.json'
        # self.convert_csv_to_json(csv_file, json_file)

        # Load JSON data
        with open(json_file, 'r') as f:
            self.data = json.load(f)

        # Inisialisasi model embedding
        self.embedding_model = SentenceTransformer('sentence-transformers/multi-qa-mpnet-base-dot-v1')

    # def convert_csv_to_json(self, csv_path, json_path):
    #     """Membaca CSV dan menyimpan sebagai JSON"""
    #     df = pd.read_csv(csv_path)
    #     df.to_json(json_path, orient='records', indent=4)

    def create_embeddings(self):
        # Membuat representasi teks gabungan dari JSON
        combined_texts = [
            f"{item['name']} {item['shop_name']} {item['main_category']} {item['sub_category']}"
            for item in self.data
        ]
        
        # Generate embeddings
        embeddings = self.embedding_model.encode(combined_texts)

        # Buat FAISS index
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings)

        # Simpan FAISS index
        faiss.write_index(index, '../Database/faiss_index.index')

        # Simpan metadata JSON
        with open('../Database/metadata.json', 'w') as f:
            json.dump(self.data, f, indent=4)

if __name__ == "__main__":
    preprocessor = Preprocessor()
    preprocessor.create_embeddings()
    print("CSV dikonversi ke JSON, embedding dibuat, dan FAISS index disimpan.")
