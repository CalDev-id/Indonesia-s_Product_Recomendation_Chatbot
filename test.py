import json
import subprocess
from bert_score import score
import argparse

def run_main_py(query, db='faiss'):
    """
    Jalankan main.py secara subprocess dan ambil output LLM response-nya.
    """
    result = subprocess.run(
        ['python', 'main.py', query, '--db', db],
        capture_output=True,
        text=True
    )
    output = result.stdout.splitlines()
    llm_response = ""
    for line in output:
        if line.startswith("LLM Response:"):
            llm_response = line.replace("LLM Response:", "").strip()
            break
    return llm_response

def evaluate(test_file_path, db='faiss'):
    with open(test_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    queries = [item['query'] for item in data]
    ground_truth = [item['ground_truth'] for item in data]
    preds = []

    print("🔍 Testing RAG results...\n")
    for item in data:
        query = item['query']
        print(f"🧠 Query: {query}")
        pred = run_main_py(query, db)
        print(f"✅ LLM Response: {pred}")
        preds.append(pred)
        print()

    # Evaluasi BERTScore
    P, R, F1 = score(preds, ground_truth, lang='id')

    print("=== 📊 Hasil Evaluasi BERTScore ===")
    for i, (query, f1) in enumerate(zip(queries, F1)):
        print(f"{i+1}. Query: {query}")
        print(f"   F1 BERTScore: {f1:.4f}")
        print()

    print("=== 📈 RATA-RATA ===")
    print(f"Precision: {P.mean():.4f}")
    print(f"Recall:    {R.mean():.4f}")
    print(f"F1 Score:  {F1.mean():.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--db', choices=['faiss', 'chroma'], default='faiss', help='Choose the database for retrieval')
    parser.add_argument('--test_file', type=str, default='test_data.json', help='Path to test JSON file')
    args = parser.parse_args()

    evaluate(args.test_file, args.db)
