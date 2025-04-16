import os
import json
import argparse  # <- tambahkan import ini

def load_test_data(test_file):
    with open(test_file, 'r') as f:
        test_file = json.load(f)
    return test_file

# main
def main():
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    parser = argparse.ArgumentParser(description="Batch test RAG chatbot with BERTScore evaluation.")
    parser.add_argument('--test_file', type=str, default='Preprocessing/ChromaDB/data_barang.json', help='Path to test JSON file')
    args = parser.parse_args()

    test_data = load_test_data(args.test_file)
    
    # Print data pertama dengan indeks 1 (elemen kedua)
    print("Data pertama [1]:", test_data[0])

if __name__ == "__main__":
    main()
