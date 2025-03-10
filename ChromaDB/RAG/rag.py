import chromadb
from dotenv import load_dotenv
from llm.groq_runtime import GroqRunTime


class Rag:
    def __init__(self):
        # setting the environment
        load_dotenv()
        DATA_PATH = r"data"
        CHROMA_PATH = r"chroma_db"

        self.chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)

        self.collection = self.chroma_client.get_or_create_collection(name="legal_doc")
    
    def rag_search(self, query):

        results = self.collection.query(
            query_texts=[query],
            n_results=1
        )

        groq_run = GroqRunTime()
        system_prompt = f"""Anda adalah asisten yang membantu. Anda menjawab pertanyaan tentang dokumen hukum di Indonesia.
Namun, Anda hanya menjawab berdasarkan pengetahuan yang saya berikan. Anda tidak menggunakan pengetahuan internal Anda dan tidak mengada-ada.
Jika Anda tidak tahu jawabannya, katakan saja: Saya tidak tahu
        --------------------
        The data:
        \n{str(results['documents'])}"""

        response = groq_run.generate_response(system_prompt, query)

        return {
            "best_match": results['documents'][0],
            "llm_response": response.choices[0].message.content
        }