import argparse
from FAISS.RAG.rag import Rag
import os

def main():

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # Set up argument parser
    parser = argparse.ArgumentParser(description="Search for product items.")
    parser.add_argument('query', type=str, help='The query to search for product items')
    args = parser.parse_args()

    # Initialize RAG
    rag = Rag()

    # Perform search using the query from arguments
    result = rag.rag_search(args.query)
    
    # Print results
    print("Best Match Resource:", result['best_match'])
    print("LLM Response:", result['llm_response'])

if __name__ == "__main__":
    main()