import argparse
import os
import json
from RAG.FaissDB.rag_faiss import RagFaiss
from RAG.ChromaDB.rag_chroma import RagChroma
from bert_score import score

# Fungsi untuk membaca file test_data.json
def load_test_data(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

# Fungsi untuk memilih RAG (Faiss atau Chroma)
def load_rag(db_choice):
    if db_choice == 'faiss':
        return RagFaiss()
    elif db_choice == 'chroma':
        return RagChroma()
    else:
        raise ValueError("Invalid database choice. Use 'faiss' or 'chroma'.")

# Fungsi untuk mengevaluasi BERTScore
def evaluate_bert_score(candidates, references):
    P, R, F1 = score(candidates, references, lang="en", verbose=False)
    return P.mean().item(), R.mean().item(), F1.mean().item(), F1

# Fungsi untuk menjalankan batch testing pada RAG
def run_batch_test(rag, test_data):
    candidates = []
    references = []

    print("🔍 Testing RAG results...\n")

    for item in test_data:
        query = item['query']
        ground_truth = item['ground_truth']

        result = rag.rag_search(query)
        llm_response = result.get('llm_response', "").strip()

        print(f"🧠 Query: {query}")
        print(f"✅ LLM Response: {llm_response if llm_response else '[EMPTY]'}\n")

        candidates.append(llm_response)
        references.append(ground_truth)

    precision, recall, avg_f1, all_f1 = evaluate_bert_score(candidates, references)

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

    # Set up argument parser
    parser = argparse.ArgumentParser(description="Search for product items using Faiss or ChromaDB.")
    parser.add_argument('db', choices=['faiss', 'chroma'], help='Choose the database for retrieval (faiss or chroma)')
    args = parser.parse_args()

    # Load test data from JSON file (hardcoded path)
    test_data_file = 'test_data.json'
    test_data = load_test_data(test_data_file)

    # Load RAG model based on chosen DB
    db_choice = args.db
    rag = load_rag(db_choice)

    # Run batch test on RAG model with test data
    run_batch_test(rag, test_data)

if __name__ == "__main__":
    main()
