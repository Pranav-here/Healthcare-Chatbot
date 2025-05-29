import streamlit as st
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
import os
import re

# Configuration
DB_FAISS_PATH = 'vectorstore/db_faiss'
HF_TOKEN = os.environ.get("HF_TOKEN")

@st.cache_resource(show_spinner=False)
def get_vector_store():
    embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    return FAISS.load_local(
        DB_FAISS_PATH,
        embedding_model,
        allow_dangerous_deserialization=True
    )

def set_custom_prompt():
    return PromptTemplate(
        template="""
<s>[INST]
You are a professional medical assistant. Your task is to answer user questions using only the context provided.

You MUST follow these rules:
- ❌ Do NOT rephrase or restate the user's question.
- ❌ Do NOT start with "The text suggests" or "Based on the context".
- ❌ Do NOT generate a summary or paraphrased version of the question.
- ❌ Do NOT include general advice, unless directly stated in the context.
- ✅ Answer directly, using only information from the context.
- ✅ Provide a clear and medically sound explanation
- ✅ Use layman's terms unless otherwise stated
- ✅ Reference which source document supports your answer
- ❌ Do NOT fabricate or assume symptoms not in the context
- ❌ Do NOT tell the user to write a diary or journal unless specifically asked
- ✅ If the context does not contain the answer, respond exactly: 
"I cannot determine from the provided information."

---

Context:
{context}

Question:
{question}
[/INST]
""",
        input_variables=["context", "question"]
    )

def load_llm():
    return HuggingFaceEndpoint(
        repo_id="HuggingFaceH4/zephyr-7b-beta",
        task="text-generation",
        temperature=0.3,
        max_new_tokens=512,
        top_p=0.95,
        repetition_penalty=1.15,
        return_full_text=False,
        huggingfacehub_api_token=HF_TOKEN
    )

def clean_text(text):
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()

def format_source_documents(docs):
    if not docs:
        return "_No supporting sources found._"

    seen = set()
    sources = []
    for i, doc in enumerate(docs, 1):
        meta = doc.metadata
        source_path = meta.get('source', 'Unknown').replace('\\', '/')
        book_name = os.path.basename(source_path).split('.pdf')[0].replace('_', ' ')
        page = meta.get('page_label') or meta.get('page', 'N/A')
        content = clean_text(doc.page_content)[:300]

        key = (book_name, page)
        if key in seen:
            continue
        seen.add(key)

        sources.append(
            f"**📖 Source {i}**\n"
            f"- **Document**: `{book_name}`\n"
            f"- **Page**: {page}\n"
            f"- **Excerpt**: {content}..."
        )
    return "\n\n".join(sources)

def render_answer(answer, sources_md):
    return f"""
### ✅ Summary:
{answer}

---

### 📚 Supporting Sources:
{sources_md}

---

⚠️ *Disclaimer: This tool is for informational purposes only. Always consult a licensed medical professional for any diagnosis or treatment.*
"""

def main():
    st.title("🩺 Ask MedicQuery")
    st.caption("Medical information assistant - Always consult a real doctor for medical decisions")

    if 'messages' not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        st.chat_message(msg['role']).markdown(msg['content'])

    if prompt := st.chat_input("Ask your medical question:"):
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({'role': 'user', 'content': prompt})

        try:
            with st.spinner("Retrieving information..."):
                vectorstore = get_vector_store()
                qa_chain = RetrievalQA.from_chain_type(
                    llm=load_llm(),
                    chain_type="stuff",
                    retriever=vectorstore.as_retriever(search_kwargs={'k': 3}),
                    return_source_documents=True,
                    chain_type_kwargs={"prompt": set_custom_prompt()}
                )

                response = qa_chain.invoke({"query": prompt})
                answer = response.get("result", "No answer returned.")
                sources_md = format_source_documents(response.get("source_documents", []))
                formatted = render_answer(answer, sources_md)

            st.chat_message("assistant").markdown(formatted)
            st.session_state.messages.append({'role': 'assistant', 'content': formatted})

        except Exception as e:
            error_msg = f"⚠️ **Error:** Something went wrong. Please try again or rephrase your question.\n\n*Details: {str(e)}*"
            st.error(error_msg)
            st.session_state.messages.append({'role': 'assistant', 'content': error_msg})

if __name__ == "__main__":
    main()
