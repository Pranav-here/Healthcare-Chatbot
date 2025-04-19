from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

'''
Initial process
    Load raw medical PDFs
    Chunk Documents
    Create Vector Embeddings
    Store Embeddings in FAISS
'''

# Step 1: Load the raw medical PDF files

path = "data/"  # folder where your PDFs are stored

# load all PDF pages from the folder
def load_pdf_files(data):
    loader = DirectoryLoader(
        data,               # path to directory
        glob='*pdf',        # only PDF files
        loader_cls=PyPDFLoader  # load one document per PDF page
    )
    doc = loader.load()  # returns list of all PDF pages as documents
    return doc


docs = load_pdf_files(data=path)
# print("Number of pages loaded:", len(docs))


# Step 2: Break the PDFs into smaller chunks

def create_chunks(extracted_data):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,     # each chunk has 500 characters
        chunk_overlap=50    # 50 characters overlap between chunks
    )
    text_chunks = text_splitter.split_documents(extracted_data)
    return text_chunks


text_chunks = create_chunks(extracted_data=docs)
# print("Number of chunks:", len(text_chunks))


# Step 3: Load the embedding model

def get_embedding_model():
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"  # small, fast model for semantic search
    )
    return embedding_model


embedding_model = get_embedding_model()


# Step 4: Create vector store using FAISS and save it

import os
DB_FAISS_PATH = os.path.join(os.path.dirname(__file__), "vectorstore", "db_faiss")  # where we want to save the database

db = FAISS.from_documents(text_chunks, embedding_model)  # convert text chunks to vectors
db.save_local(DB_FAISS_PATH)  # save the FAISS index to disk
