import streamlit as st
from PyPDF2 import PdfReader
from datetime import datetime
import asyncio

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate


# Fix event loop issue (for Streamlit)
try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())


# -----------------------------
# PDF TEXT EXTRACTION
# -----------------------------
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
    return text


# -----------------------------
# TEXT CHUNKING
# -----------------------------
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    return text_splitter.split_text(text)


# -----------------------------
# LOAD EMBEDDINGS (Cached)
# -----------------------------
@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


# -----------------------------
# CREATE VECTOR STORE
# -----------------------------
def get_vector_store(text_chunks):
    embeddings = load_embeddings()
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    return vector_store


# -----------------------------
# CREATE QA CHAIN
# -----------------------------
def get_conversational_chain(api_key):
    prompt_template = """
    Answer the question using the provided context only.
    If the answer is not in the context, say:
    "Answer is not available in the context."

    Context:
    {context}

    Question:
    {question}

    Answer:
    """

    model = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        temperature=0.3,
        google_api_key=api_key
    )

    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )

    chain = load_qa_chain(
        model,
        chain_type="stuff",
        prompt=prompt
    )

    return chain


# -----------------------------
# HANDLE USER QUESTION
# -----------------------------
def user_input(user_question, api_key):
    if not api_key:
        st.warning("Please enter your Google Gemini API Key.")
        return

    if "vectorstore" not in st.session_state or st.session_state.vectorstore is None:
        st.warning("Please upload and process PDFs first.")
        return

    vectorstore = st.session_state.vectorstore
    docs = vectorstore.similarity_search(user_question)

    chain = get_conversational_chain(api_key)

    response = chain(
        {"input_documents": docs, "question": user_question},
        return_only_outputs=True
    )

    answer = response["output_text"]

    st.session_state.conversation_history.append((
        user_question,
        answer,
        datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ))

    st.write("### 🧠 Answer")
    st.write(answer)


# -----------------------------
# MAIN APP
# -----------------------------
def main():
    st.set_page_config(page_title="Chat with Multiple PDFs", page_icon="📚")
    st.header("📚 Chat with Multiple PDFs")

    if "conversation_history" not in st.session_state:
        st.session_state.conversation_history = []

    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None

    st.sidebar.title("Settings")

    api_key = st.sidebar.text_input(
        "Enter your Google Gemini API Key",
        type="password"
    )

    pdf_docs = st.sidebar.file_uploader(
        "Upload PDF files",
        accept_multiple_files=True
    )

    if st.sidebar.button("Process PDFs"):
        if not pdf_docs:
            st.warning("Please upload at least one PDF.")
        else:
            with st.spinner("Processing PDFs..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                vectorstore = get_vector_store(text_chunks)
                st.session_state.vectorstore = vectorstore
            st.success("PDFs processed and indexed successfully!")

    user_question = st.text_input(
        "Ask a question about your PDFs",
        disabled=st.session_state.vectorstore is None
    )

    if user_question:
        user_input(user_question, api_key)

    if st.session_state.conversation_history:
        st.write("### 📜 Conversation History")

        for q, a, t in reversed(st.session_state.conversation_history):
            st.markdown(f"**🕒 {t}**")
            st.markdown(f"**You:** {q}")
            st.markdown(f"**AI:** {a}")
            st.markdown("---")


if __name__ == "__main__":
    main()