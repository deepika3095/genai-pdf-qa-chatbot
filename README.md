## Development of a PDF-Based Question-Answering Chatbot Using LangChain

### AIM :
To design and implement a question-answering chatbot capable of processing and extracting information from a provided PDF document using LangChain, and to evaluate its effectiveness by testing its responses to diverse queries derived from the document's content.

### PROBLEM STATEMENT :
The objective is to build a chatbot that answers questions from the content of a PDF document. Using LangChain and OpenAI, the system processes the PDF, retrieves relevant information, and provides accurate responses to user queries.

### DESIGN STEPS :
#### STEP 1 :
Load and Process PDF – Import the PDF document, extract its content, and split it into smaller text chunks for efficient handling.
#### STEP 2 :
Embed and Store Content – Convert text chunks into embeddings and store them in an in-memory vector database for fast retrieval.
#### STEP 3 :
Build Question-Answering System – Use LangChain’s RetrievalQA with an OpenAI model to retrieve relevant chunks and generate accurate answers to user queries.

### PROGRAM :
```
#NAME: DEEPIKA R
#REG: 212223230038

import os
import openai
import sys
sys.path.append('../..')

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())
openai.api_key = os.environ['OPENAI_API_KEY']

from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import DocArrayInMemorySearch
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
def build_pdf_qa(pdf_path: str):
    if not os.path.isfile(pdf_path):
        raise FileNotFoundError(f"File not found: {pdf_path}")
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_documents(docs)
    db = DocArrayInMemorySearch.from_documents(chunks, OpenAIEmbeddings())
    return RetrievalQA.from_chain_type(
        llm=ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0),
        retriever=db.as_retriever()
    )
pdf_file = "docs/cs229_lectures/pdf-3.pdf"
qa = build_pdf_qa(pdf_file)
questions = [
    "What is Perceptron Learning Algorithm in this document?",
    "Explain the main topics covered in this PDF?",
    "What are the key formulas mentioned in this document?"
]
for i, q in enumerate(questions, 1):
    answer = qa.run(q)
    print(f"Q{i}: {q}")
    print(f"A{i}: {answer}\n")
loader = PyPDFLoader(pdf_file)
pages = loader.load()
print(f"Loaded {len(pages)} pages from the PDF.")

```
### OUTPUT :
<img width="1290" height="403" alt="image" src="https://github.com/user-attachments/assets/202b9d73-7deb-4f92-bb0f-38170c383756" />

### RESULT :
Thus, a question-answering chatbot capable of processing and extracting information from a provided PDF document using LangChain is executed successfully.
