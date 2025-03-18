import argparse
from RAG.rag_faiss import RagFaiss
import os

def main():

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # Set up argument parser
    parser = argparse.ArgumentParser(description="Search for product items.")
    parser.add_argument('query', type=str, help='The query to search for product items')
    # parser.add_argument('-v', '--version', choices=['1', '2'], default=1, help='Model Version')
    args = parser.parse_args()

    config = vars(args)

    # Get arguments values
    query = config['query']
    # version = int(config['version'])

    rag = RagFaiss()

    result = rag.rag_search(query)
    # if version == 1:
    #     result = rag.rag_search(query)
    # else:
    #     return
    
    # Print results
    print("Best Match Resource:", result['best_match'])
    print("LLM Response:", result['llm_response'])

if __name__ == "__main__":
    main()