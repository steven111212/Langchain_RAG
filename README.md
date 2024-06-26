# RAG (Retrieval-Augmented Generation) Project

This project implements two types of Retrieval-Augmented Generation (RAG) systems: text_RAG and PDF_RAG. These systems are designed to enhance question-answering capabilities by leveraging large language models and efficient information retrieval techniques.

## Features

1. text_RAG: 
   - Processes plain text documents
   - Uses FAISS for efficient text retrieval
   - Employs OpenAI's language models for question answering

2. PDF_RAG:
   - Processes PDF documents, including text, tables, and images
   - Extracts and analyzes images from PDFs
   - Handles table structures in PDFs
   - Provides more comprehensive analysis of complex documents

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
```
### PDF_RAG

For processing PDF documents:

```python
from pdf_rag import PDF_RAG

# Initialize PDF_RAG
pdf_rag = PDF_RAG("your-api-key", "path/to/your/pdf/file.pdf", 
                  infer_table_structure=True, extract_images_in_pdf=True)

# Ask a question
answer, relevant_images = pdf_rag.user_ask("What is the main topic of this PDF?")
print(answer)

# If you want to see information about relevant images
for i, image in enumerate(relevant_images):
    print(f"Relevant image {i + 1}: {image}")
```



