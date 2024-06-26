from langchain_openai import OpenAI
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
import os

class text_RAG:
    def __init__(self, api_key, file_path):
        
        os.environ["OPENAI_API_KEY"] = api_key

        with open(file_path, 'r') as file:
            text = file.read()

        text_splitter = RecursiveCharacterTextSplitter(separators=["\n\n", "\n"], chunk_size=4000, chunk_overlap=350)
        all_splits = text_splitter.split_text(text)

        embedding = OpenAIEmbeddings()
        self.db = FAISS.from_texts(all_splits, embedding)


        self.llm = OpenAI(temperature=0.2)


    def user_ask(self, prompt):

        docs = self.db.similarity_search(prompt)
        answer = self.llm(input_documents=docs, question = prompt)

        return answer
    


if __name__ == '__main__':

    test = text_RAG("your_openai_api_key", 'good.txt')
    test.user_ask('what is this text talking about briefly describe')

