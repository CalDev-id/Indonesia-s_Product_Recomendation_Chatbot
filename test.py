import argparse
import os
import json
from RAG.FaissDB.rag_faiss import RagFaiss
from RAG.ChromaDB.rag_chroma import RagChroma
from bert_score import score

def load_rag(db_choice):
    return RagFaiss() if db_choice == 'faiss' else RagChroma()

def evaluate_bert_score(candidates, references):
    P, R, F1 = score(candidates, references, lang="en", verbose=False)
    return P.mean().item(), R.mean().item(), F1.mean().item(), F1

def run_batch_test(test_file, rag):
    with open(test_file, 'r') as f:
        test_data = json.load(f)

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

    # Modify the argument parser to support --test_file
    parser = argparse.ArgumentParser(description="Batch test RAG chatbot with BERTScore evaluation.")
    parser.add_argument('--test_file', type=str, default='test_data.json', help='Path to test JSON file')
    parser.add_argument('--db', choices=['faiss', 'chroma'], default='faiss', help='Choose database (faiss or chroma)')
    args = parser.parse_args()

    rag = load_rag(args.db)
    run_batch_test(args.test_file, rag)

if __name__ == "__main__":
    main()
