#  MedicQuery - A Medical RAG Chatbot

MedicQuery Bot is an intelligent Retrieval-Augmented Generation (RAG) chatbot built using LangChain, FAISS, HuggingFace models, and Streamlit. It allows users to ask medical questions and receive answers strictly based on medical textbooks loaded into the system.

⚠️ *Note: This chatbot is for educational and informational purposes only. It does not provide real medical advice.*

---

##  Features

-  Loads and parses medical PDF textbooks
-  Splits content into context-rich chunks
-  Embeds and indexes documents using FAISS
-  Integrates HuggingFace’s `Mistral-7B-Instruct-v0.3` for LLM responses
-  Offers a clean chat UI via Streamlit
-  Displays source documents used for each answer

---

##  Tech Stack

- Python 3.11
- LangChain
- FAISS
- HuggingFace Transformers
- Streamlit
- PyPDFLoader

---

## 🗂 Project Structure

```
.
├── data/                       # PDF medical textbooks
├── vectorstore/               # FAISS index and vectors
├── medical-chatbot.py                     # Streamlit frontend
├── create_memory_for_llm.py   # PDF loader, chunker, embedding generator
├── connect_memory_with_llm.py # CLI-based query pipeline
├── requirements.txt / Pipfile # Python dependencies
└── README.md                  # This file
```

---

##  Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/pranav-here/HealthcareBot.git
cd HealthcareBot
```

### 2. Install Dependencies

Using pipenv:

```bash
pipenv install
pipenv shell
```

Or using pip:

```bash
pip install -r requirements.txt
```

### 3. Add Environment Variable

Create a `.env` file in the root directory:

```env
HF_TOKEN=your_huggingface_api_key
```

Make sure your HuggingFace account has access to the model you're using.

---

##  Adding PDFs

Place all your medical PDF files inside the `data/` folder. Each page will be processed and embedded individually.

---

##  Creating Vector Embeddings

Run the following script to load, chunk, and embed your PDFs:

```bash
python create_memory_for_llm.py
```

---

##  Query from Terminal

```bash
python connect_memory_with_llm.py
```

---

##  Launch the Chat UI

```bash
streamlit run app.py
```

---

##  Sample Prompt

```text
What is a cough and what causes it?
```

---

##  Disclaimer

> This tool is built using public medical content and open-source models for **educational purposes only**. It does **not** replace professional medical advice, diagnosis, or treatment. Always consult with a qualified healthcare provider.

---

##  Future Enhancements

- [ ] Multi-language support
- [ ] Better Streamlit UI
- [ ] Answer summarization
- [ ] Query history

---

##  License

MIT License

---

##  Acknowledgements

- LangChain
- HuggingFace
- FAISS
