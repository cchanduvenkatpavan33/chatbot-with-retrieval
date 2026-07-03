import os
import streamlit as st
import google.generativeai as genai
from dotenv import load_dotenv

from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    Docx2txtLoader,
)

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# ======================================================
# PAGE CONFIG
# ======================================================

st.set_page_config(
    page_title="AI Retrieval Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ======================================================
# LOAD ENVIRONMENT
# ======================================================

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    st.error("❌ GOOGLE_API_KEY not found.")
    st.stop()

genai.configure(api_key=GOOGLE_API_KEY)

model = genai.GenerativeModel("gemini-2.5-flash")

# ======================================================
# SESSION STATE
# ======================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "vector_db" not in st.session_state:
    st.session_state.vector_db = None

# ======================================================
# CUSTOM CSS
# ======================================================

st.markdown(
    """
<style>

.block-container{
    padding-top:1.5rem;
}

#MainMenu{
visibility:hidden;
}

footer{
visibility:hidden;
}

header{
visibility:hidden;
}

.stButton>button{
width:100%;
border-radius:12px;
height:48px;
font-weight:bold;
}

.chat-title{
font-size:42px;
font-weight:700;
text-align:center;
margin-bottom:5px;
}

.chat-subtitle{
text-align:center;
color:gray;
margin-bottom:25px;
}

</style>
""",
    unsafe_allow_html=True,
)

# ======================================================
# SIDEBAR
# ======================================================

with st.sidebar:

    st.image("assets/logo.png", width=90)

    st.title("AI Retrieval Assistant")

    st.success("🟢 Ready")

    st.divider()

    st.markdown("### 🚀 Technology")

    st.markdown("""
- Gemini 2.5 Flash
- LangChain
- HuggingFace
- FAISS
- Streamlit
""")

    st.divider()

    uploaded_files = st.file_uploader(
        "📂 Upload Documents",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True,
    )

# ======================================================
# MAIN PAGE
# ======================================================

st.markdown(
    "<div class='chat-title'>🤖 AI Retrieval Assistant</div>",
    unsafe_allow_html=True,
)

st.markdown(
    "<div class='chat-subtitle'>Chat with your documents using Gemini + RAG</div>",
    unsafe_allow_html=True,
)

st.divider()
# ======================================================
# SAVE DOCUMENTS
# ======================================================

if uploaded_files:

    os.makedirs("documents", exist_ok=True)

    for uploaded_file in uploaded_files:

        save_path = os.path.join(
            "documents",
            uploaded_file.name
        )

        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

    st.success(f"✅ {len(uploaded_files)} file(s) uploaded successfully.")

    if st.button("🚀 Build Knowledge Base"):

        with st.spinner("Building Vector Database..."):

            documents = []

            for filename in os.listdir("documents"):

                filepath = os.path.join(
                    "documents",
                    filename
                )

                try:

                    if filename.lower().endswith(".txt"):

                        loader = TextLoader(
                            filepath,
                            encoding="utf-8"
                        )

                    elif filename.lower().endswith(".pdf"):

                        loader = PyPDFLoader(filepath)

                    elif filename.lower().endswith(".docx"):

                        loader = Docx2txtLoader(filepath)

                    else:
                        continue

                    documents.extend(loader.load())

                except Exception as e:

                    st.warning(f"Skipping {filename}")

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )

            chunks = splitter.split_documents(documents)

            embedding = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )

            db = FAISS.from_documents(
                chunks,
                embedding
            )

            db.save_local("vectorstore")

            st.session_state.vector_db = db

        st.success("✅ Knowledge Base Created Successfully!")

        st.balloons()
        # ======================================================
# LOAD VECTOR DATABASE
# ======================================================

if st.session_state.vector_db is None:

    if os.path.exists("vectorstore"):

        embedding = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        st.session_state.vector_db = FAISS.load_local(
            "vectorstore",
            embedding,
            allow_dangerous_deserialization=True
        )       
# ======================================================
# CHAT INTERFACE
# ======================================================

db = st.session_state.vector_db

if db is None:

    st.info("""
📂 Upload one or more PDF, DOCX or TXT files.

Then click

🚀 Build Knowledge Base

to begin chatting with your documents.
""")

    st.stop()

# ======================================================
# DISPLAY CHAT HISTORY
# ======================================================

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])

# ======================================================
# CHAT INPUT
# ======================================================

question = st.chat_input("💬 Ask a question about your documents...")

if question:

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )

    with st.chat_message("user"):

        st.markdown(question)

    with st.chat_message("assistant"):

        with st.spinner("🤖 Thinking..."):

            docs = db.similarity_search(
                question,
                k=4
            )

            context = "\n\n".join(
                doc.page_content
                for doc in docs
            )

            prompt = f"""
You are an AI assistant.

Use ONLY the context below.

If the answer is not available, say:

"I couldn't find that information in the uploaded documents."

Context:

{context}

Question:

{question}

Answer:
"""

            try:

                response = model.generate_content(prompt)

                answer = response.text

            except Exception as e:

                answer = f"❌ {e}"

            st.markdown(answer)

            with st.expander("📚 Retrieved Sources"):

                for i, doc in enumerate(docs, start=1):

                    st.markdown(f"### Source {i}")

                    st.write(doc.page_content)

                    st.divider()

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer
        }
    )
# ======================================================
# SIDEBAR UTILITIES
# ======================================================

with st.sidebar:

    st.divider()

    st.subheader("📊 Statistics")

    st.metric("Uploaded Files", len(uploaded_files) if uploaded_files else 0)

    st.metric("Messages", len(st.session_state.messages))

    st.divider()

    if st.button("🧹 Clear Chat"):

        st.session_state.messages = []

        st.rerun()

    if st.session_state.messages:

        chat_text = ""

        for msg in st.session_state.messages:

            role = msg["role"].capitalize()

            chat_text += f"{role}: {msg['content']}\n\n"

        st.download_button(
            label="📥 Download Chat",
            data=chat_text,
            file_name="chat_history.txt",
            mime="text/plain",
        )

    st.divider()

    st.info(
        """
### 💡 Suggested Questions

• Summarize this document

• Give an abstract

• What is the conclusion?

• Explain the methodology

• List the important points

• Compare the topics

• What are the advantages?

• Give a short summary
"""
    )

# ======================================================
# FOOTER
# ======================================================

st.divider()

st.markdown(
    """
<div style="text-align:center;color:gray;font-size:15px;">

Made with ❤️ using

<b>Google Gemini 2.5 Flash • LangChain • FAISS • Streamlit</b>

</div>
""",
    unsafe_allow_html=True,
)            