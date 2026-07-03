import streamlit as st
import os
from dotenv import load_dotenv

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

import google.generativeai as genai

# ---------------------------------------
# Load Environment Variables
# ---------------------------------------

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    st.error("❌ GOOGLE_API_KEY not found inside .env file")
    st.stop()

genai.configure(api_key=GOOGLE_API_KEY)

model = genai.GenerativeModel("gemini-2.5-flash")

# ---------------------------------------
# Page Configuration
# ---------------------------------------

st.set_page_config(
    page_title="AI Retrieval Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------
# Custom CSS
# ---------------------------------------

st.markdown("""
<style>

html, body, [class*="css"]{
    background:#0E1117;
    color:white;
    font-family:Segoe UI;
}

/* Hide Streamlit Menu */

#MainMenu{
visibility:hidden;
}

footer{
visibility:hidden;
}

header{
visibility:hidden;
}

/* Sidebar */

section[data-testid="stSidebar"]{

background:#1A1C24;

}

/* Buttons */

.stButton>button{

width:100%;

height:55px;

border-radius:12px;

background:#4F46E5;

color:white;

font-size:18px;

font-weight:bold;

border:none;

transition:0.3s;

}

.stButton>button:hover{

background:#6D5EF9;

transform:scale(1.02);

}

/* Text Area */

textarea{

border-radius:12px !important;

}

/* Chat Card */

.chat-card{

background:#1A1C24;

padding:20px;

border-radius:15px;

margin-bottom:15px;

border:1px solid #30363D;

}

/* Title */

.big-title{

text-align:center;

font-size:58px;

font-weight:800;

color:white;

}

.subtitle{

text-align:center;

font-size:24px;

color:#A0AEC0;

margin-bottom:30px;

}

</style>

""", unsafe_allow_html=True)
# ============================================================
# Sidebar
# ============================================================

with st.sidebar:

    st.markdown("## 🤖 AI Retrieval Assistant")

    st.success("🟢 System Ready")

    st.divider()

    st.markdown("### ⚙️ Technology")

    st.markdown("""
✅ Gemini 2.5 Flash

✅ LangChain

✅ HuggingFace

✅ FAISS

✅ Streamlit
""")

    st.divider()

    st.info("""
This chatbot uses **Retrieval-Augmented Generation (RAG)**.

It first searches your uploaded documents and then Gemini generates the final answer.
""")

    st.divider()

    uploaded_files = st.file_uploader(
        "📂 Upload Documents",
        type=["txt", "pdf", "docx"],
        accept_multiple_files=True
    )

    if uploaded_files:

        os.makedirs("documents", exist_ok=True)

        uploaded = 0

        for file in uploaded_files:

            save_path = os.path.join("documents", file.name)

            with open(save_path, "wb") as f:
                f.write(file.read())

            uploaded += 1

        st.success(f"✅ {uploaded} file(s) uploaded")

        if st.button("🔄 Rebuild Knowledge Base"):

            with st.spinner("Building Vector Database..."):

                os.system("python ingest.py")

            st.success("Knowledge Base Updated!")

    st.divider()

    if st.button("🗑 Clear Chat"):

        st.session_state.messages = []

        st.rerun()
        # ============================================================
# Chat History
# ============================================================

if "messages" not in st.session_state:
    st.session_state.messages = []
    # ============================================================
# Load Vector Database
# ============================================================

embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

try:

    db = FAISS.load_local(
        "vectorstore",
        embedding,
        allow_dangerous_deserialization=True
    )

except:

    st.warning("⚠ Vector database not found.")

    st.stop()
    # ============================================================
# Main Page
# ============================================================

st.markdown(
    "<h1 class='big-title'>🤖 AI Retrieval Assistant</h1>",
    unsafe_allow_html=True
)

st.markdown(
    "<p class='subtitle'>Chat with your documents using Gemini 2.5 Flash + RAG</p>",
    unsafe_allow_html=True
)

st.divider()

# ============================================================
# User Question
# ============================================================

question = st.chat_input("Ask anything about your documents...")

if question:

    # Save user message
    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )

    # Display user message
    with st.chat_message("user"):
        st.markdown(question)

    # Search FAISS
    docs = db.similarity_search(question, k=4)

    context = "\n\n".join([doc.page_content for doc in docs])

    prompt = f"""
You are an intelligent AI assistant.

Answer ONLY using the information provided below.

If the answer is not available in the documents, politely say:

"I couldn't find that information in the uploaded documents."

Context:

{context}

Question:

{question}

Provide a clear, professional answer.
"""

    with st.chat_message("assistant"):

        with st.spinner("🤖 Gemini is thinking..."):

            try:

                response = model.generate_content(prompt)

                answer = response.text

            except Exception as e:

                answer = f"❌ Error:\n\n{e}"

        st.markdown(answer)

        with st.expander("📚 Retrieved Sources"):

            for i, doc in enumerate(docs):

                st.markdown(f"### Source {i+1}")

                st.write(doc.page_content)

                st.divider()

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer
        }
    )
    # ============================================================
# Display Previous Chat History
# ============================================================

if len(st.session_state.messages) > 0:

    st.divider()

    st.subheader("💬 Conversation")

    for message in st.session_state.messages:

        with st.chat_message(message["role"]):
            st.markdown(message["content"])

else:

    st.divider()

    st.info("""
👋 **Welcome!**

Upload your TXT, PDF, or DOCX documents using the sidebar.

Then ask questions like:

• What is Artificial Intelligence?

• Summarize this document.

• Explain Machine Learning.

• List the important points.

Your assistant will search your documents first and then use Gemini to generate an accurate answer.
""")
# ============================================================
# Download Conversation
# ============================================================

if st.session_state.messages:

    conversation = ""

    for msg in st.session_state.messages:

        role = msg["role"].upper()

        conversation += f"{role}\n"

        conversation += msg["content"]

        conversation += "\n\n"

    st.download_button(

        "📥 Download Conversation",

        conversation,

        file_name="conversation.txt",

        mime="text/plain"

    )
    # ============================================================
# Footer
# ============================================================

st.divider()

st.markdown(
"""
<div style='text-align:center;color:gray;'>

### 🚀 AI Retrieval Assistant

Built with ❤️ using

**Gemini 2.5 Flash • LangChain • HuggingFace • FAISS • Streamlit**

Version 2.0

</div>
""",
unsafe_allow_html=True
)
# ============================================================
# Chat Statistics
# ============================================================

with st.sidebar:

    st.divider()

    st.markdown("### 📊 Statistics")

    total_messages = len(st.session_state.messages)

    user_questions = sum(
        1 for m in st.session_state.messages
        if m["role"] == "user"
    )

    ai_answers = sum(
        1 for m in st.session_state.messages
        if m["role"] == "assistant"
    )

    st.metric("Messages", total_messages)
    st.metric("Questions", user_questions)
    st.metric("Answers", ai_answers)