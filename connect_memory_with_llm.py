import os
from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

import warnings
warnings.filterwarnings("ignore")  # ignore warnings in output

# get Hugging Face token from environment
HF_TOKEN = os.environ.get("HF_TOKEN")
# choose the LLM model to use
huggingface_repo_id = "mistralai/Mistral-7B-Instruct-v0.3"

# load the LLM (text generator) from Hugging Face
def load_llm(huggingface_repo_id):
    llm = HuggingFaceEndpoint(
        repo_id=huggingface_repo_id,
        task="text-generation",
        temperature=0.5,
        huggingfacehub_api_token=HF_TOKEN,  # pass token here
        model_kwargs={"max_length": 512}  # limit response length
    )
    return llm

# path to the saved vector database
DF_FAISS_PATH = "vectorstore/db_faiss"

# prompt template for how the bot should talk
custom_prompt_template = """
You are a highly intelligent and helpful AI medical assistant. Your job is to answer user queries based strictly on the context provided below.

Follow these rules:
- Use only the information provided in the context to form your answer.
- If the answer cannot be determined from the context, respond with: "I'm not sure based on the provided information."
- Do NOT fabricate or infer facts that are not explicitly present.
- Be clear, direct, and avoid unnecessary small talk.
- Format your answer in complete, professional sentences.

Context: {context}

Question: {question}

Answer:
"""

# turn the above text into a usable prompt object
def set_custom_prompt(custom_prompt_template):
    prompt = PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])
    return prompt

# load embedding model to convert text into vectors
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# load existing FAISS vector store (already built from documents)
db = FAISS.load_local(DF_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)

# build the chatbot using the LLM + vector retriever + custom prompt
qa_chain = RetrievalQA.from_chain_type(
    llm=load_llm(huggingface_repo_id),
    chain_type="stuff",
    retriever=db.as_retriever(search_kwargs={'k': 3}),  # get top 3 most similar chunks
    return_source_documents=True,
    chain_type_kwargs={'prompt': set_custom_prompt(custom_prompt_template)}
)

# take a question from the user
user_query = input("Write Query here: ")
# run the bot with the user question
response = qa_chain.invoke({'query': user_query})

# show the answer
print("Result:", response["result"])
# show the documents it used to answer
print("Source Documents:", response["source_documents"])
