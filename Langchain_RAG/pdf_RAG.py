from typing import Any
from pydantic import BaseModel
from unstructured.partition.pdf import partition_pdf
import uuid
import os
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.schema.messages import HumanMessage, SystemMessage
from langchain.schema.document import Document
from langchain_community.vectorstores import FAISS
from langchain.retrievers.multi_vector import MultiVectorRetriever
import base64

class PDF_RAG:
    def __init__(self, api_key, file_path, infer_table_structure = True, extract_images_in_pdf = True) -> None:
        
        os.environ["OPENAI_API_KEY"] = api_key
        path = './figures'
        raw_pdf_elements = partition_pdf(
        filename= file_path,
        # Using pdf format to find embedded image blocks
        extract_images_in_pdf=extract_images_in_pdf,
        # Use layout model (YOLOX) to get bounding boxes (for tables) and find titles
        # Titles are any sub-section of the document
        infer_table_structure=infer_table_structure,
        # Post processing to aggregate text once we have the title
        chunking_strategy="by_title",
        # Chunking params to aggregate text blocks
        # Attempt to create a new chunk 3800 chars
        # Attempt to keep chunks > 2000 chars
        # Hard max on chunks
        max_characters=4000,
        new_after_n_chars=3800,
        combine_text_under_n_chars=2000,
    )
        #raw_pdf_elements = extract_pdf_elements('./figures', file_path, infer_table_structure, extract_images_in_pdf)

        # Create a dictionary to store counts of each type
        text_elements = []
        table_elements = []

        text_summaries = []
        table_summaries = []
        summary_prompt = """
        Summarize the following {element_type}:
        {element}
        """
        summary_chain = LLMChain(
        llm=ChatOpenAI(model="gpt-3.5-turbo", max_tokens=1024), 
        prompt=PromptTemplate.from_template(summary_prompt)
        )

        for e in raw_pdf_elements:
            if 'CompositeElement' in repr(e):
                text_elements.append(e.text)
                summary = summary_chain.run({'element_type':'text', 'element': e})
                text_summaries.append(summary)
            elif ' Table' in repr(e):
                table_elements.append(e.text)
                summary = summary_chain.run({'element_type': 'table', 'element': e})
                table_summaries.append(summary)  

        image_elements = []
        image_summaries = []
        if extract_images_in_pdf:
            for i in os.listdir(path):
                if i.endswith(('.png', '.jpg', '.jpeg')):
                    image_path = os.path.join(path, i)
                    encoded_image = encode_image(image_path)
                    image_elements.append(encoded_image)
                    summary = summarize_image(encoded_image)
                    image_summaries.append(summary)

        # Create Documents and Vectorstore
        documents = []
        retrieve_contents = []

        for e, s in zip(text_elements, text_summaries):
            i = str(uuid.uuid4())
            doc = Document(
                page_content = s,
                metadata = {
                    'id': i,
                    'type': 'text',
                    'original_content': e
                }
            ) 
            retrieve_contents.append((i, e))
            documents.append(doc)
            
        for e, s in zip(table_elements, table_summaries):
            doc = Document(
                page_content = s,
                metadata = {
                    'id': i,
                    'type': 'table',
                    'original_content': e
                }
            )
            retrieve_contents.append((i, e))
            documents.append(doc)
            
        for e, s in zip(image_elements, image_summaries):
            doc = Document(
                page_content = s,
                metadata = {
                    'id': i,
                    'type': 'image',
                    'original_content': e
                }
            )
            retrieve_contents.append((i, s))
            documents.append(doc)

        self.vectorstore = FAISS.from_documents(documents=documents, embedding=OpenAIEmbeddings())

        answer_template = """
        Answer the question based only on the following context, which can include text, images and tables:
        {context}
        Question: {question} 
        """
        self.answer_chain = LLMChain(llm=ChatOpenAI(model="gpt-3.5-turbo", max_tokens=1024), prompt=PromptTemplate.from_template(answer_template))


    def user_ask(self, prompt):
        relevant_docs = self.vectorstore.similarity_search(prompt)
        context = ""
        relevant_images = []
        for d in relevant_docs:
            if d.metadata['type'] == 'text':
                context += '[text]' + d.metadata['original_content']
            elif d.metadata['type'] == 'table':
                context += '[table]' + d.metadata['original_content']
            elif d.metadata['type'] == 'image':
                context += '[image]' + d.page_content
                relevant_images.append(d.metadata['original_content'])
        result = self.answer_chain.run({'context': context, 'question': prompt})
        return result, relevant_images



def encode_image(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode('utf-8')
    
def summarize_image(encoded_image):
    prompt = [
        SystemMessage(content="You are a bot that is good at analyzing images."),
        HumanMessage(content=[
            {
                "type": "text", 
                "text": "Describe the contents of this image."
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{encoded_image}"
                },
            },
        ])
    ]
    response = ChatOpenAI(model="gpt-4o", max_tokens=1024).invoke(prompt)
    return response.content


# Get elements
# Extract elements from PDF
# def extract_pdf_elements(path, fname, infer_table_structure = False, extract_images_in_pdf = False):
#     """
#     Extract images, tables, and chunk text from a PDF file.
#     path: File path, which is used to dump images (.jpg)
#     fname: File name
#     """
#     return partition_pdf(
#         filename= fname,
#         extract_images_in_pdf=extract_images_in_pdf,
#         infer_table_structure=infer_table_structure,
#         chunking_strategy="by_title",
#         max_characters=4000,
#         new_after_n_chars=3800,
#         combine_text_under_n_chars=2000,
#         image_output_dir_path=path,
#     )



if __name__ == '__main__':

    test = PDF_RAG("your_openai_api_key", './data/Swin.pdf')
    result, relevant_images = test.user_ask('What is the difference between Swim Transformer and ViT')
    print(result)
    print(relevant_images)

