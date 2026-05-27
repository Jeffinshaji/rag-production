import streamlit as st
import requests
import uuid
import os
from dotenv import load_dotenv

load_dotenv()

# ── Config ─────────────────────────────────────────────────────
FUNCTION_BASE_URL = os.environ.get(
    "FUNCTION_BASE_URL",
    "https://rag-jeff-pipeline.azurewebsites.net/api"
)

UPLOAD_URL    = f"{FUNCTION_BASE_URL}/upload"
CHAT_URL      = f"{FUNCTION_BASE_URL}/chat"
DOCUMENTS_URL = f"{FUNCTION_BASE_URL}/documents"

# ── Page Config ────────────────────────────────────────────────
st.set_page_config(
    page_title = "RAG Chatbot",
    page_icon  = "🤖",
    layout     = "wide"
)

# ── Custom CSS ─────────────────────────────────────────────────
st.markdown("""
<style>
    /* Main background */
    .stApp {
        background-color: #0f1117;
        color: #ffffff;
    }

    /* Chat message - user */
    .user-message {
        background-color: #1e3a5f;
        padding: 12px 16px;
        border-radius: 18px 18px 4px 18px;
        margin: 8px 0;
        margin-left: 20%;
        color: white;
    }

    /* Chat message - bot */
    .bot-message {
        background-color: #1a1a2e;
        border: 1px solid #2d2d44;
        padding: 12px 16px;
        border-radius: 18px 18px 18px 4px;
        margin: 8px 0;
        margin-right: 20%;
        color: white;
    }

    /* Upload area */
    .upload-box {
        background-color: #1a1a2e;
        border: 2px dashed #3d3d5c;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
    }

    /* Success message */
    .success-box {
        background-color: #0d3320;
        border: 1px solid #1a6640;
        border-radius: 8px;
        padding: 12px;
        color: #4ade80;
    }

    /* Header */
    .main-header {
        font-size: 2rem;
        font-weight: bold;
        color: #60a5fa;
        margin-bottom: 0;
    }

    .sub-header {
        color: #94a3b8;
        font-size: 0.9rem;
        margin-top: 0;
    }

    /* Document badge */
    .doc-badge {
        background-color: #1e3a5f;
        border-radius: 20px;
        padding: 4px 12px;
        font-size: 0.8rem;
        color: #93c5fd;
        display: inline-block;
        margin: 4px;
    }

    /* Divider */
    hr {
        border-color: #2d2d44;
    }
</style>
""", unsafe_allow_html=True)


# ── Session State ──────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "uploaded_docs" not in st.session_state:
    st.session_state.uploaded_docs = []


# ── Helper Functions ───────────────────────────────────────────
def upload_pdf(file):
    """Upload PDF to Azure Function"""
    try:
        response = requests.post(
            UPLOAD_URL,
            files   = {"file": (file.name, file.getvalue(), "application/pdf")},
            timeout = 120
        )
        return response.json()
    except requests.exceptions.Timeout:
        return {"error": "Upload timed out. Please try again."}
    except Exception as e:
        return {"error": str(e)}


def ask_question(question):
    """Send question to Azure Function"""
    try:
        response = requests.post(
            CHAT_URL,
            json    = {
                "question"  : question,
                "session_id": st.session_state.session_id
            },
            timeout = 30
        )
        return response.json()
    except requests.exceptions.Timeout:
        return {"error": "Request timed out. Please try again."}
    except Exception as e:
        return {"error": str(e)}


def get_documents():
    """Get list of uploaded documents"""
    try:
        response = requests.get(DOCUMENTS_URL, timeout=10)
        return response.json().get("documents", [])
    except:
        return []


# ══════════════════════════════════════════════════════
# LAYOUT — TWO COLUMNS
# Left: Upload + Documents
# Right: Chat
# ══════════════════════════════════════════════════════

# Header
st.markdown('<p class="main-header">🤖 RAG Chatbot</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Upload your documents and ask questions</p>', unsafe_allow_html=True)
st.divider()

# Two columns
left_col, right_col = st.columns([1, 2])


# ── LEFT COLUMN — Upload + Documents ──────────────────────────
with left_col:

    # Upload Section
    st.markdown("### 📄 Upload Document")
    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type    = ["pdf"],
        help    = "Upload any PDF document to chat with it"
    )

    if uploaded_file:
        if st.button("🚀 Upload & Process", use_container_width=True):
            with st.spinner("Uploading and processing... This may take 1-2 minutes..."):
                result = upload_pdf(uploaded_file)

            if "error" in result:
                st.error(f"❌ {result['error']}")
            else:
                st.success(f"✅ {result.get('message', 'Uploaded successfully!')}")
                st.info(f"📊 Created {result.get('chunks', 0)} searchable chunks")
                if uploaded_file.name not in st.session_state.uploaded_docs:
                    st.session_state.uploaded_docs.append(uploaded_file.name)

    st.divider()

    # Documents Section
    st.markdown("### 📚 Uploaded Documents")

    if st.button("🔄 Refresh", use_container_width=True):
        st.session_state.uploaded_docs = get_documents()

    docs = get_documents()
    if docs:
        for doc in docs:
            st.markdown(
                f'<span class="doc-badge">📄 {doc}</span>',
                unsafe_allow_html=True
            )
    else:
        st.caption("No documents uploaded yet")

    st.divider()

    # Clear Chat
    st.markdown("### ⚙️ Settings")
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages   = []
        st.session_state.session_id = str(uuid.uuid4())
        st.rerun()

    st.caption(f"Session: {st.session_state.session_id[:8]}...")


# ── RIGHT COLUMN — Chat ────────────────────────────────────────
with right_col:
    st.markdown("### 💬 Chat")

    # Chat history container
    chat_container = st.container(height=500)

    with chat_container:
        if not st.session_state.messages:
            st.markdown("""
            <div style='text-align:center; color:#4a5568; padding:40px;'>
                <div style='font-size:3rem;'>💬</div>
                <div style='font-size:1.1rem; margin-top:10px;'>
                    Upload a PDF and start asking questions!
                </div>
            </div>
            """, unsafe_allow_html=True)

        for message in st.session_state.messages:
            if message["role"] == "user":
                st.markdown(
                    f'<div class="user-message">👤 {message["content"]}</div>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f'<div class="bot-message">🤖 {message["content"]}</div>',
                    unsafe_allow_html=True
                )

    # Chat Input
    st.markdown("")
    col1, col2 = st.columns([5, 1])

    with col1:
        question = st.text_input(
            "Ask a question",
            placeholder = "e.g. What is Data Science?",
            label_visibility = "collapsed"
        )

    with col2:
        send_btn = st.button("Send 📤", use_container_width=True)

    # Handle send
    if send_btn and question:
        # Add user message
        st.session_state.messages.append({
            "role"   : "user",
            "content": question
        })

        # Get answer
        with st.spinner("🔍 Searching documents..."):
            result = ask_question(question)

        if "error" in result:
            answer = f"⚠️ Error: {result['error']}"
        else:
            answer = result.get("answer", "No answer received.")

        # Add bot message
        st.session_state.messages.append({
            "role"   : "assistant",
            "content": answer
        })

        st.rerun()

    # Suggested questions
    if not st.session_state.messages:
        st.markdown("**💡 Try asking:**")
        suggestions = [
            "What is this document about?",
            "Summarize the key points",
            "What are the main topics covered?"
        ]
        cols = st.columns(3)
        for i, suggestion in enumerate(suggestions):
            with cols[i]:
                if st.button(suggestion, use_container_width=True):
                    st.session_state.messages.append({
                        "role"   : "user",
                        "content": suggestion
                    })
                    with st.spinner("Thinking..."):
                        result = ask_question(suggestion)
                    answer = result.get("answer", "No answer received.")
                    st.session_state.messages.append({
                        "role"   : "assistant",
                        "content": answer
                    })
                    st.rerun()
