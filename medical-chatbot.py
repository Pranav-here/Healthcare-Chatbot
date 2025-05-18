import streamlit as st
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
import os
from langchain_huggingface import HuggingFaceEndpoint
from transformers import pipeline
from langchain.llms import HuggingFacePipeline
import re

# path to saved vector DB
DB_FAISS_PATH = 'vectorstore/db_faiss'
# load token from environment variable
HF_TOKEN = os.environ.get("HF_TOKEN")


# load the saved FAISS vector DB with embedding model
@st.cache_resource
def get_vector_store():
    embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    return db


# make a prompt template using Langchain format
def set_custom_prompt(custom_prompt_template):
    prompt = PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])
    return prompt

def load_llm_local_cpu():
    pipe = pipeline(
        "text-generation",
        model="mistralai/Mistral-7B-Instruct-v0.3",
        device=-1  # Force CPU
    )
    llm = HuggingFacePipeline(pipeline=pipe)
    return llm

# load the Mistral model from Hugging Face
def load_llm(huggingface_repo_id, HF_TOKEN):
    llm = HuggingFaceEndpoint(
        repo_id=huggingface_repo_id,
        task="text-generation",
        temperature=0.5,
        huggingfacehub_api_token=HF_TOKEN,
        model_kwargs={"max_length": 512, "device_map": "cpu"}
    )
    return llm


# format source documents for clean display in chat
def clean_text(text):
    """Clean up bad characters from scanned PDFs."""
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)  # remove non-ASCII
    text = re.sub(r'\s+', ' ', text).strip()   # collapse extra whitespace
    return text

def format_source_documents(docs):
    formatted = []
    for i, doc in enumerate(docs, 1):
        # Source metadata
        source_path = doc.metadata.get('source', 'Unknown Source')
        book_name = source_path.split('/')[-1].split('\\')[-1].split('.pdf')[0].replace('_', ' ')
        page_number = doc.metadata.get('page_label', doc.metadata.get('page', 'N/A'))

        # Content cleanup
        content = clean_text(doc.page_content)
        if len(content) > 300:
            content = content[:300] + '...'

        # Markdown formatting
        formatted.append(
            f"### 📚 Source {i}\n"
            f"**Book:** *{book_name}*\n\n"
            f"**Page:** {page_number}\n\n"
            f"**Excerpt:**\n> {content}\n"
        )

    return "\n---\n".join(formatted)


# main Streamlit app
def main():
    st.title("️Ask MedicQuery!🩺🧑‍⚕")  # app title

    # initialize message history
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    # show past messages
    for message in st.session_state.messages:
        st.chat_message(message['role']).markdown(message['content'])

    # input box for the user
    prompt = st.chat_input('Pass your prompt here: ')
    if prompt:
        # display and store user message
        st.chat_message('user').markdown(prompt)
        st.session_state.messages.append({'role': 'user', 'content': prompt})

        # chatbot prompt logic
        custom_prompt_template = """You are a highly intelligent and helpful AI medical assistant. Your job is to answer user queries based strictly on the context provided below.

        Follow these rules:
        - Use only the information provided in the context to form your answer.
        - If the answer cannot be determined from the context, respond with: "I'm not sure based on the provided information."
        - Do NOT fabricate or infer facts that are not explicitly present.
        - Be clear, direct, and avoid unnecessary small talk.
        - Format your answer in complete, professional sentences.

        Context: {context}

        Question: {question}

        Answer:"""

        huggingface_repo_id = "mistralai/Mistral-7B-Instruct-v0.3"

        try:
            # load vector database
            vectorstore = get_vector_store()

            # build the QA chain
            qa_chain = RetrievalQA.from_chain_type(
                # llm=load_llm(huggingface_repo_id=huggingface_repo_id, HF_TOKEN=HF_TOKEN),
                llm = load_llm_local_cpu(),
                chain_type="stuff",
                retriever=vectorstore.as_retriever(search_kwargs={'k': 3}),
                return_source_documents=True,
                chain_type_kwargs={'prompt': set_custom_prompt(custom_prompt_template)}
            )

            # get the answer from the bot
            response = qa_chain.invoke({'query': prompt})

            # format the output
            answer = response['result']
            sources = format_source_documents(response["source_documents"])

            formatted_response = (
                f"**Answer:**\n{answer}\n\n"
                f"---\n"
                f"⚠️ *Disclaimer: This response is based on the provided material and should not be taken as medical advice. Please consult a licensed healthcare professional for real-world concerns.*\n\n"
                f"---\n"
                f"**Supporting Sources:**\n"
                f"{sources}"
            )

            # display bot message and store it
            st.chat_message('assistant').markdown(formatted_response)
            st.session_state.messages.append({'role': 'assistant', 'content': formatted_response})

        except Exception as e:
            st.error(f"Error: {str(e)}")


# run the app
if __name__ == "__main__":
    main()
