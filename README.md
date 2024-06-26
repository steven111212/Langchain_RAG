# RAG (Retrieval-Augmented Generation) Project

This project implements two types of Retrieval-Augmented Generation (RAG) systems: text_RAG and PDF_RAG. These systems are designed to enhance question-answering capabilities by leveraging large language models and efficient information retrieval techniques.

## Features

- text_RAG: A RAG system for plain text documents
- PDF_RAG: An advanced RAG system capable of processing PDF documents, including text, tables, and images

- ## Usage

### text_RAG

The text_RAG class is designed for processing plain text documents:

```python
from text_rag import text_RAG

# Initialize the text_RAG system
rag = text_RAG("your-api-key", "path/to/your/text/file.txt")

# Ask a question
answer = rag.user_ask("What is this text talking about?")
print(answer)

- ## Usage

### RDF_RAG

The PDF_RAG class is designed for processing plain text documents:

```python
from pdf_rag import PDF_RAG

# Initialize the PDF_RAG system  
rag = PDF_RAG("your-api-key", "path/to/your/pdf/file.pdf")

# Ask a question
answer, relevant_images = rag.user_ask("What is the main topic of this PDF?")
print(answer)
# The relevant_images variable will contain any images related to the answer
