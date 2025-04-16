import argparse
import os
import json
from RAG.FaissDB.rag_faiss import RagFaiss
from RAG.ChromaDB.rag_chroma import RagChroma
from bert_score import score
from LLM.groq_runtime import GroqRunTime


class Test:
    def __init__(self, db_choice, test_file):
        self.test_file = test_file
        self.db_choice = db_choice
        self.rag = self.load_rag()

    def load_test_data(self):
        with open(self.test_file, 'r') as f:
            return json.load(f)

    def load_rag(self):
        return RagFaiss() if self.db_choice == 'faiss' else RagChroma()

    def get_refined_query(self, product):
        """ Menggunakan LLM untuk membuat query untuk test pencarian produk """
        groq_run = GroqRunTime()
        system_prompt = (
            "Anda adalah asisten pembuat kalimat pencarian barang berdasarkan deskripsi berbahasa Indonesia. "
            "Tolong buat kalimat pencarian untuk mencari produk dari data produk berikut:\n"
            f"Judul Produk: {product['title']}\n"
            f"Deskripsi: {product['description']}\n"
            f"Kategori: {product['categories']}\n"
            "Contoh kata pencariannya: saya mencari produk skincare dengan spf 50 yang bagus untuk kulit berminyak\n"
            "langsung jawab saja tanpa ada kata lainnya seperti (berikut adalah..)"
        )
        response = groq_run.generate_response(system_prompt, "")
        return response.choices[0].message.content.strip()

    def make_ground_truth(self, product):
        """ Gabungkan title, description, categories sebagai ground truth """
        return f"{product['title']}. {product['description']} Kategori: {product['categories']}"

    def create_refined_test_data(self, output_path="refined_test_data.json", max_data=10):
        test_data = self.load_test_data()[:max_data]  # Ambil hanya 10 data pertama
        refined_data = []

        print("🛠 Membuat refined query dan ground truth untuk 10 data pertama...\n")

        for item in test_data:
            refined_query = self.get_refined_query(item)
            ground_truth = self.make_ground_truth(item)

            refined_data.append({
                "query": refined_query,
                "ground_truth": ground_truth
            })

        with open(output_path, 'w') as f:
            json.dump(refined_data, f, indent=2)

        print(f"✅ Refined test data disimpan ke {output_path}")

    def evaluate_bert_score(self, candidates, references):
        P, R, F1 = score(candidates, references, lang="en", verbose=False)
        return P.mean().item(), R.mean().item(), F1.mean().item(), F1

    def run_batch_test(self, refined_file="refined_test_data.json"):
        with open(refined_file, 'r') as f:
            test_data = json.load(f)

        candidates = []
        references = []

        print("🔍 Testing RAG results...\n")

        for item in test_data:
            query = item['query']
            ground_truth = item['ground_truth']

            result = self.rag.rag_search(query)
            llm_response = result.get('llm_response', "").strip()

            print(f"🧠 Query: {query}")
            print(f"✅ LLM Response: {llm_response if llm_response else '[EMPTY]'}\n")

            candidates.append(llm_response)
            references.append(ground_truth)

        precision, recall, avg_f1, all_f1 = self.evaluate_bert_score(candidates, references)

        print("=== 📊 Hasil Evaluasi BERTScore ===")
        for i, (item, f1_score) in enumerate(zip(test_data, all_f1), 1):
            print(f"{i}. Query: {item['query']}")
            print(f"   F1 BERTScore: {f1_score.item():.4f}\n")

        print("=== 📈 RATA-RATA ===")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1 Score:  {avg_f1:.4f}")


def main():
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    parser = argparse.ArgumentParser(description="Batch test RAG chatbot with BERTScore evaluation.")
    parser.add_argument('--test_file', type=str, default='Preprocessing/ChromaDB/data_barang.json', help='Path to test JSON file')
    parser.add_argument('--db', choices=['faiss', 'chroma'], default='faiss', help='Choose database (faiss or chroma)')
    args = parser.parse_args()

    refined_file_path = "refined_test_data.json"
    test = Test(db_choice=args.db, test_file=args.test_file)

    if not os.path.exists(refined_file_path):
        test.create_refined_test_data(output_path=refined_file_path, max_data=10)

    test.run_batch_test(refined_file=refined_file_path)


if __name__ == "__main__":
    main()
