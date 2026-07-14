import streamlit as st
import os
import tempfile
import functools
from concurrent.futures import ThreadPoolExecutor, as_completed
from pinecone import Pinecone, ServerlessSpec
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

# Page Configuration
st.set_page_config(
    page_title="YCE AI Dashboard",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Premium Styling (Covers both RAG and Chatbot components)
st.markdown("""
<style>
    /* Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');
    
    /* Global Styles */
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    /* ── MAIN PAGE TEXT VISIBILITY ─────────────────────────── */
    
    /* Base body text */
    .stApp, .main .block-container {
        color: #e2e8f0 !important;
    }
    
    /* All plain text, paragraphs, spans */
    .stApp p, .stApp span, .stApp div,
    .stMarkdown p, .stMarkdown li, .stMarkdown span {
        color: #e2e8f0 !important;
    }
    
    /* Headings */
    .stApp h1, .stApp h2, .stApp h3,
    .stApp h4, .stApp h5, .stApp h6 {
        color: #ffffff !important;
    }
    
    /* Widget Labels (selectbox, slider, text input etc.) */
    .stApp label,
    .stApp .stWidgetLabel,
    .stApp span[data-testid="stWidgetLabel"],
    .stApp .stWidgetLabel p {
        color: #c8c3d4 !important;
        font-weight: 500 !important;
    }
    
    /* Selectbox & input text */
    .stApp .stSelectbox div[data-baseweb="select"] span,
    .stApp .stTextInput input,
    .stApp .stTextArea textarea {
        color: #f0ecff !important;
    }
    
    /* Tab labels */
    .stTabs [data-baseweb="tab"] {
        color: #a099c0 !important;
        font-weight: 500;
    }
    .stTabs [aria-selected="true"] {
        color: #ffffff !important;
        border-bottom: 2px solid #ff007f !important;
    }
    
    /* Expander label text */
    .streamlit-expanderHeader,
    .streamlit-expanderHeader p,
    details > summary,
    details > summary span {
        color: #e2e8f0 !important;
        font-weight: 600 !important;
    }
    
    /* Metric labels and values */
    [data-testid="stMetricLabel"],
    [data-testid="stMetricValue"] {
        color: #ffffff !important;
    }
    
    /* Info / Warning / Success / Error boxes */
    .stAlert p,
    .stAlert div {
        color: #1a1a2e !important;
    }
    
    /* File uploader text */
    .stFileUploader label,
    .stFileUploader section p,
    .stFileUploader span {
        color: #c8c3d4 !important;
    }
    
    /* Slider thumb label */
    .stSlider [data-testid="stTickBarMin"],
    .stSlider [data-testid="stTickBarMax"],
    .stSlider p {
        color: #a099c0 !important;
    }
    
    /* General small helper/caption text */
    small, .caption, .stCaption {
        color: #9e99b0 !important;
    }
    
    /* Code blocks */
    code, pre {
        color: #00f2fe !important;
        background: rgba(0, 242, 254, 0.06) !important;
        border-radius: 6px;
    }
    /* ── END MAIN PAGE TEXT VISIBILITY ──────────────────────── */
    
    /* Main Background Aurora */
    .stApp {
        background: radial-gradient(circle at 80% 20%, #1c0d3a 0%, #0a0915 50%, #05040b 100%) !important;
        background-attachment: fixed !important;
    }
    
    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: rgba(13, 10, 30, 0.95) !important;
        border-right: 1px solid rgba(138, 43, 226, 0.15) !important;
    }
    
    /* Sidebar Text Visibility Fix */
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] h4,
    section[data-testid="stSidebar"] span[data-testid="stWidgetLabel"],
    section[data-testid="stSidebar"] .stMarkdown p {
        color: #cbd5e1 !important;
    }
    
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] h4,
    section[data-testid="stSidebar"] strong {
        color: #ffffff !important;
    }
    
    /* Header Gradient styling */
    .dashboard-header {
        background: linear-gradient(135deg, rgba(30, 15, 60, 0.3) 0%, rgba(10, 8, 25, 0.6) 100%);
        padding: 2.2rem;
        border-radius: 20px;
        margin-bottom: 2rem;
        border: 1px solid rgba(255, 0, 127, 0.2);
        box-shadow: 0 8px 32px 0 rgba(255, 0, 127, 0.08), inset 0 0 20px rgba(138, 43, 226, 0.15);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
    }
    
    .dashboard-title {
        background: linear-gradient(90deg, #ff007f 0%, #a020f0 40%, #00f2fe 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.8rem;
        font-weight: 700;
        margin: 0;
        letter-spacing: -0.5px;
    }
    
    .dashboard-subtitle {
        color: #c7c2d1;
        font-size: 1.05rem;
        margin-top: 0.5rem;
    }
    
    /* RAG Result Card Styling */
    .result-card {
        background: rgba(22, 16, 45, 0.45);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(255, 255, 255, 0.04);
        border-left: 4px solid #00f2fe;
        border-radius: 16px;
        padding: 1.5rem;
        margin-bottom: 1.2rem;
        transition: all 0.4s cubic-bezier(0.165, 0.84, 0.44, 1);
        box-shadow: 0 4px 24px rgba(0, 0, 0, 0.25);
    }
    
    .result-card:hover {
        transform: translateY(-4px);
        border-color: rgba(255, 0, 127, 0.5);
        border-left: 4px solid #ff007f;
        box-shadow: 0 12px 36px rgba(255, 0, 127, 0.15);
    }
    
    .card-meta {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 1rem;
        font-size: 0.85rem;
    }
    
    .meta-badge-score {
        background: linear-gradient(90deg, #00f2fe 0%, #4facfe 100%);
        color: #0a0915;
        font-weight: 700;
        padding: 0.25rem 0.75rem;
        border-radius: 30px;
        box-shadow: 0 0 10px rgba(0, 242, 254, 0.3);
    }
    
    .meta-badge-page {
        background: rgba(255, 0, 127, 0.15);
        color: #ff5aa7;
        border: 1px solid rgba(255, 0, 127, 0.3);
        font-weight: 600;
        padding: 0.25rem 0.75rem;
        border-radius: 30px;
    }
    
    .card-filename {
        color: #f1f5f9;
        font-weight: 600;
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
        max-width: 60%;
    }
    
    .card-text {
        color: #e2e8f0;
        font-size: 0.95rem;
        line-height: 1.6;
    }
    
    /* Message Bubbles custom styling */
    .stChatMessage {
        border-radius: 16px !important;
        margin-bottom: 1.2rem !important;
        padding: 1rem 1.4rem !important;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1) !important;
    }
    
    /* User Chat Bubble (Pink/Indigo Gradient) */
    .stChatMessage[data-testid="stChatMessageUser"] {
        background: linear-gradient(135deg, #7928ca 0%, #ff007f 100%) !important;
        border: 1px solid rgba(255, 0, 127, 0.2) !important;
        color: #ffffff !important;
    }
    
    /* Assistant Chat Bubble (Neon-Glow Glass) */
    .stChatMessage[data-testid="stChatMessageAssistant"] {
        background: rgba(20, 15, 40, 0.65) !important;
        backdrop-filter: blur(10px) !important;
        border: 1px solid rgba(255, 255, 255, 0.04) !important;
        border-left: 4px solid #00f2fe !important;
    }
    
    /* Chat input custom focus styling */
    .stChatInput textarea {
        background-color: rgba(15, 10, 30, 0.8) !important;
        border-color: rgba(138, 43, 226, 0.3) !important;
        box-shadow: 0 0 10px rgba(138, 43, 226, 0.1) !important;
        border-radius: 12px !important;
        color: #ffffff !important;
        transition: all 0.3s ease;
    }
    
    .stChatInput textarea:focus {
        border-color: #ff007f !important;
        box-shadow: 0 0 15px rgba(255, 0, 127, 0.25) !important;
    }
    
    /* Custom Scrollbars */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    ::-webkit-scrollbar-track {
        background: rgba(10, 8, 20, 0.5);
    }
    ::-webkit-scrollbar-thumb {
        background: rgba(138, 43, 226, 0.4);
        border-radius: 4px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: rgba(255, 0, 127, 0.6);
    }
</style>
""", unsafe_allow_html=True)

# ── Credentials: loaded from Streamlit Secrets (never hardcode in source) ──
# Locally: set in .streamlit/secrets.toml
# On Streamlit Cloud: set via the Secrets manager in the dashboard
DEFAULT_PINECONE_API_KEY = st.secrets.get("PINECONE_API_KEY", "")
DEFAULT_GEMINI_API_KEY   = st.secrets.get("GEMINI_API_KEY", "")
INDEX_NAME               = st.secrets.get("PINECONE_INDEX", "yce")

if not DEFAULT_PINECONE_API_KEY or not DEFAULT_GEMINI_API_KEY:
    st.error(
        "🔑 **API keys not configured.**\n\n"
        "Add `PINECONE_API_KEY`, `GEMINI_API_KEY`, and `PINECONE_INDEX` to your "
        "[Streamlit Secrets](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/secrets-management)."
    )
    st.stop()

# Initialize Session State Variables (so they persist across mode switching)
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar Navigation
st.sidebar.markdown("## 🧭 Navigation")
app_mode = st.sidebar.selectbox(
    "Select Application Mode",
    options=["🔍 YCE Virtual RAG Search", "💬 Gemini Chatbot"],
    index=0
)

# Load Embedding Model Globally
@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

embeddings = load_embeddings()

# Initialize Pinecone Client
@st.cache_resource
def get_pinecone_client(api_key):
    try:
        return Pinecone(api_key=api_key)
    except Exception as e:
        return None

# ── INTENT-AWARE ADAPTIVE RAG RETRIEVAL ─────────────────────────────────────

import re as _re

_TOC_PATTERNS = [
    "table of contents", "list sections", "what sections",
    "what topics", "what chapters", "contents page",
    "show index", "document index", "list of sections",
    "what does the document cover", "what is covered",
    "sections in", "chapters in", "topics in", "headings",
    "show me the structure", "document structure",
]
_SUMMARIZE_PATTERNS = [
    "summarise", "summarize", "summary", "give me a summary",
    "explain section", "what is in section", "brief on", "overview of",
    "describe section", "what does section", "tell me about section",
    "give overview", "outline section",
]
_FULL_DOC_PATTERNS = [
    "summarise the document", "summarize the document",
    "summarise the report",  "summarize the report",
    "summarise this report", "summarize this report",
    "overall summary", "executive summary", "full summary",
    "summarise everything", "summarize everything",
    "all sections", "all section", "list all sections",
    "summarise all", "summarize all",
    "entire document", "entire report", "whole document", "whole report",
    "what is the document about", "what is this report about",
    "give me the full", "give me a complete summary",
    "summarise the whole", "summarize the whole",
]


def detect_query_intent(query: str) -> str:
    """
    Classify query to pick the right retrieval strategy.
    Returns: 'toc' | 'summarize_section' | 'summarize_full' | 'general'
    """
    q = query.lower().strip()
    # Check full-doc patterns first (higher priority)
    if any(p in q for p in _FULL_DOC_PATTERNS):
        return "summarize_full"
    if any(p in q for p in _TOC_PATTERNS):
        return "toc"
    if any(p in q for p in _SUMMARIZE_PATTERNS):
        # If just 'summarise'/'summarize' with no section reference after it,
        # treat as full-doc summary
        stripped = q
        for trigger in ["summarise", "summarize", "summary"]:
            stripped = stripped.replace(trigger, "").strip()
        if len(stripped) < 4:   # nothing meaningful left = full doc
            return "summarize_full"
        return "summarize_section"
    return "general"


def extract_section_reference(query: str) -> str:
    """Pull out the section identifier from a summarize query."""
    q = query.lower()
    for trigger in _SUMMARIZE_PATTERNS:
        q = q.replace(trigger, "")
    # also strip "section", "chapter", "part"
    q = _re.sub(r'\b(section|chapter|part|the|of|about)\b', ' ', q)
    q = _re.sub(r'\s+', ' ', q).strip(" .,;:")
    return q if q else query


def generate_query_variants(query: str, intent: str = "general") -> list[str]:
    """Generate retrieval queries adapted to intent."""
    q = query.strip().rstrip("?.!")

    if intent == "toc":
        return [
            "table of contents",
            "list of sections chapters",
            "document outline index",
            "contents page headings",
            f"{q} table of contents sections",
            "methodology components analysis",
            "section 1 section 2 section 3",
        ]

    if intent == "summarize_section":
        section = extract_section_reference(query)
        return [
            section,
            f"section {section}",
            f"{section} content analysis findings",
            f"{section} data results details",
            f"{section} report discussion",
            f"details of {section}",
            query,
        ]

    if intent == "summarize_full":
        return [
            "executive summary overview",
            "key findings conclusions recommendations",
            "financial analysis results",
            "audit observations",
            "summary of the report",
            "overall assessment",
            query,
        ]

    # General fact query — domain-agnostic variants
    q_base = query.strip().rstrip("?.!")
    return [
        query,
        f"What is {q_base}?",
        f"Tell me about {q_base}",
        f"{q_base} details information data",
        f"{q_base} as documented in the report",
    ][:5]


# Domain keywords used to auto-detect document type from namespace name
_DOMAIN_HINTS = {
    "financial": ["audit", "finance", "budget", "account", "balance", "ledger",
                  "revenue", "expense", "income", "profit", "loss", "tax",
                  "invoice", "payment", "cash", "bank", "fiscal", "economic",
                  "statement", "annual", "quarter", "pkr", "usd", "eur"],
    "legal":     ["contract", "agreement", "legal", "law", "clause", "terms",
                  "policy", "regulation", "compliance", "act", "court",
                  "litigation", "statute", "rights", "obligation", "liability"],
    "hr":        ["hr", "human", "resource", "employee", "staff", "salary",
                  "payroll", "leave", "attendance", "performance", "appraisal",
                  "recruitment", "onboarding", "training", "benefits"],
    "technical": ["technical", "engineering", "software", "system", "architecture",
                  "specification", "design", "api", "database", "infrastructure",
                  "network", "server", "code", "module", "requirement"],
    "medical":   ["medical", "clinical", "patient", "diagnosis", "treatment",
                  "drug", "hospital", "health", "disease", "therapy", "protocol"],
    "academic":  ["research", "study", "thesis", "dissertation", "paper",
                  "abstract", "methodology", "hypothesis", "literature", "citation"],
}

def _detect_document_domain(namespace: str) -> str:
    """Infer document type from namespace name for better HyDE generation."""
    ns_lower = namespace.lower()
    scores = {domain: 0 for domain in _DOMAIN_HINTS}
    for domain, keywords in _DOMAIN_HINTS.items():
        for kw in keywords:
            if kw in ns_lower:
                scores[domain] += 1
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "general"


_DOMAIN_DESCRIPTIONS = {
    "financial": "financial report, balance sheet, or audit document with monetary figures",
    "legal":     "legal contract, agreement, or policy document with clauses and obligations",
    "hr":        "human resources document with employee data, policies, or procedures",
    "technical": "technical specification, software design, or engineering document",
    "medical":   "medical or clinical report with patient data and treatment information",
    "academic":  "academic research paper with methodology, findings, and references",
    "general":   "professional document with structured sections and factual content",
}


def safe_llm_call(
    selected_model: str,
    api_key: str,
    messages: list,
    temperature: float = 0.2,
    max_output_tokens: int = 900,
):
    """
    Executes a non-streaming Gemini API call with auto-fallback for 503/429 errors.
    Order: selected_model -> gemini-2.5-flash -> gemini-2.0-flash
    """
    models_to_try = [selected_model]
    for fallback in ["gemini-1.5-flash", "gemini-2.0-flash"]:
        if fallback not in models_to_try:
            models_to_try.append(fallback)
            
    last_err = None
    for model in models_to_try:
        try:
            llm = ChatGoogleGenerativeAI(
                model=model,
                temperature=temperature,
                google_api_key=api_key,
                max_output_tokens=max_output_tokens,
            )
            return llm.invoke(messages), model
        except Exception as e:
            err_str = str(e).lower()
            if any(term in err_str for term in ["503", "429", "quota", "limit", "unavailable"]):
                last_err = e
                continue
            raise e
    raise last_err


def safe_llm_stream(
    selected_model: str,
    api_key: str,
    messages: list,
    temperature: float = 0.2,
):
    """
    Generates a stream from Gemini API with auto-fallback for initial 503/429 errors.
    """
    models_to_try = [selected_model]
    for fallback in ["gemini-1.5-flash", "gemini-2.0-flash"]:
        if fallback not in models_to_try:
            models_to_try.append(fallback)
            
    for model in models_to_try:
        try:
            llm = ChatGoogleGenerativeAI(
                model=model,
                temperature=temperature,
                google_api_key=api_key
            )
            stream = llm.stream(messages)
            iterator = iter(stream)
            try:
                first_chunk = next(iterator)
            except StopIteration:
                # empty stream
                return iter([]), model
            
            # Reassemble generator to yield peeked item first
            def stream_generator():
                yield first_chunk
                for chunk in iterator:
                    yield chunk
            return stream_generator(), model
        except Exception as e:
            err_str = str(e).lower()
            if any(term in err_str for term in ["503", "429", "quota", "limit", "unavailable"]):
                continue
            raise e
            
    # Final fallback if all failed (to throw the error properly)
    llm = ChatGoogleGenerativeAI(
        model=models_to_try[-1],
        temperature=temperature,
        google_api_key=api_key
    )
    return llm.stream(messages), models_to_try[-1]


@functools.lru_cache(maxsize=128)
def generate_hyde_passage(
    query: str,
    gemini_api_key: str,
    model_name: str,
    intent: str = "general",
    namespace: str = "",
) -> str:
    """
    HyDE: Generate a realistic document excerpt whose embedding
    is closer to actual chunks than the raw query.
    Cached with lru_cache so the same query never calls Gemini twice.
    max_output_tokens capped at 80 for speed (shorter is still semantically rich).
    """
    try:
        domain = _detect_document_domain(namespace)
        doc_desc = _DOMAIN_DESCRIPTIONS.get(domain, _DOMAIN_DESCRIPTIONS["general"])

        hyde_llm = ChatGoogleGenerativeAI(
            model=model_name,
            temperature=0.0,
            google_api_key=gemini_api_key,
            max_output_tokens=80,   # short = fast, still semantically rich
        )
        if intent == "toc":
            hyde_system = (
                f"You are a document index simulator for a {doc_desc}. "
                "Output a realistic numbered 'Table of Contents' with 6-8 sections. "
                "No preamble, just the TOC text."
            )
        elif intent in ("summarize_section", "summarize_full"):
            hyde_system = (
                f"You are a document excerpt simulator for a {doc_desc}. "
                "Write 2-3 declarative sentences summarising the topic. "
                "Output raw document text only."
            )
        else:
            hyde_system = (
                f"You are a document excerpt simulator for a {doc_desc}. "
                "Write 1-2 sentences that would appear verbatim in this document "
                "and directly answer the question. Raw text only, no preamble."
            )
            
        messages = [
            SystemMessage(content=hyde_system),
            HumanMessage(content=f"Q: {query}")
        ]
        
        response, _model_used = safe_llm_call(
            selected_model=model_name,
            api_key=gemini_api_key,
            messages=messages,
            temperature=0.0,
            max_output_tokens=80
        )
        passage = response.content.strip()
        return passage if passage else query
    except Exception:
        return query   # silent fallback — retrieval continues without HyDE



def _pinecone_search(vector_store, q_variant: str, k: int) -> list:
    """Single Pinecone search call — designed to be run in a thread."""
    try:
        return vector_store.similarity_search_with_score(q_variant, k=k)
    except Exception:
        return []


def _broad_document_sweep(vector_store, k_per_query: int = 2) -> list:
    """
    Retrieve chunks from DIVERSE parts of the document using 6 parallel keyword sweeps.
    Each sweep retrieves k_per_query=2 chunks → ~12 unique chunks total.
    All sweeps fire simultaneously via ThreadPoolExecutor.
    """
    sweep_queries = [
        "introduction background purpose scope objectives",
        "methodology approach findings results analysis",
        "observations issues recommendations actions",
        "financial income revenue expenditure balance",
        "conclusion summary highlights key points",
        "management response annexures appendix",
    ]
    seen: dict[str, tuple] = {}
    with ThreadPoolExecutor(max_workers=6) as executor:
        futures = [
            executor.submit(_pinecone_search, vector_store, q, k_per_query)
            for q in sweep_queries
        ]
        for fut in as_completed(futures):
            for doc, score in fut.result():
                key = doc.page_content[:200]
                if key not in seen:
                    seen[key] = (doc, score)
    chunks = list(seen.values())
    chunks.sort(key=lambda x: x[0].metadata.get("page", 0))   # page order
    return chunks


def _fast_summarize(
    chunks: list,
    gemini_api_key: str,
    model_name: str,
    domain_desc: str = "document",
) -> str:
    """
    Fast single-call document summarization.
    No per-chunk Gemini calls (those were the bottleneck).
    Strategy:
      1. Truncate each chunk to 250 chars — keeps key facts, loses padding
      2. Concatenate with page labels into one compact context (~3000 chars)
      3. ONE Gemini call to produce a structured executive summary
    Total latency: ~2-4 seconds instead of 25+ seconds.
    """
    if not chunks:
        return "No document content was retrieved to summarize."

    # Build compact context: 250 chars per chunk, labelled by page
    MAX_PER_CHUNK = 250
    MAX_TOTAL = 5000
    lines, total = [], 0
    for doc, _score in chunks:
        page = doc.metadata.get("page", 0) + 1
        snippet = doc.page_content[:MAX_PER_CHUNK].replace("\n", " ").strip()
        line = f"[P{page}] {snippet}"
        if total + len(line) > MAX_TOTAL:
            break
        lines.append(line)
        total += len(line)
    context = "\n".join(lines)

    llm = ChatGoogleGenerativeAI(
        model=model_name,
        temperature=0.2,
        google_api_key=gemini_api_key,
        max_output_tokens=900,
    )
    system = (
        f"You are a professional analyst summarising a {domain_desc}. "
        "The context below contains short excerpts from different pages of the document. "
        "Write a well-structured executive summary under these exact headings:\n"
        "**🎯 Purpose & Scope** | **🔍 Methodology** | **📊 Key Findings** | "
        "**💰 Financial Highlights** | **⚠️ Observations & Issues** | **✅ Recommendations**\n"
        "Use bullet points. Include all specific figures, dates, and names you find. "
        "If a section has no data in context, write 'Not covered in retrieved pages'."
    )
    messages = [
        SystemMessage(content=system),
        HumanMessage(content=f"Document excerpts:\n{context}")
    ]
    try:
        response, _model_used = safe_llm_call(
            selected_model=model_name,
            api_key=gemini_api_key,
            messages=messages,
            temperature=0.2,
            max_output_tokens=900
        )
        return response.content.strip()
    except Exception as e:
        err = str(e)
        if "429" in err or "quota" in err.lower() or "limit" in err.lower():
            return (
                "🚫 **Rate limit hit during summarization.**\n\n"
                "**Condensed document excerpts (raw):**\n\n" + context
            )
        return f"**Summary generation failed:** {err}\n\n**Raw excerpts:**\n{context}"


def adaptive_retrieve(
    vector_store,
    query: str,
    k: int,
    gemini_api_key: str = "",
    model_name: str = "gemini-1.5-flash",
    namespace: str = "",
    speed_mode: bool = False,
) -> tuple[list, str]:
    """
    Intent-aware retrieval dispatcher. Works for ANY document type.

    summarize_full    → broad sweep of 10 diverse queries (no HyDE) + map-reduce handled externally
    summarize_section → targeted section queries, HyDE only if not speed_mode
    toc               → TOC-targeted queries
    general           → parallel paraphrase + HyDE (if not speed_mode)

    All Pinecone queries run in parallel via ThreadPoolExecutor.
    Returns (results, intent_label).
    """
    intent = detect_query_intent(query)

    if intent == "summarize_full":
        # Broad sweep — no HyDE, 6 parallel queries, 2 chunks each → ~12 unique chunks
        chunks = _broad_document_sweep(vector_store, k_per_query=2)
        return chunks, intent

    # Adaptive K for other intents
    effective_k = k
    if intent == "toc":
        effective_k = max(k, 8)
    elif intent == "summarize_section":
        effective_k = max(k, 6)

    queries_to_run = generate_query_variants(query, intent)

    # HyDE: cached; skipped for speed_mode or summarize_full
    if not speed_mode and gemini_api_key and intent not in ("summarize_full",):
        hyde_passage = generate_hyde_passage(
            query, gemini_api_key, model_name, intent, namespace
        )
        if hyde_passage.lower() != query.lower():
            queries_to_run.append(hyde_passage)

    # ⚡ Parallel Pinecone searches
    seen_content: dict[str, tuple] = {}
    with ThreadPoolExecutor(max_workers=min(len(queries_to_run), 6)) as executor:
        futures = {
            executor.submit(_pinecone_search, vector_store, q, effective_k): q
            for q in queries_to_run
        }
        for future in as_completed(futures):
            for doc, score in future.result():
                key = doc.page_content[:200]
                if key not in seen_content or score > seen_content[key][1]:
                    seen_content[key] = (doc, score)

    merged = sorted(seen_content.values(), key=lambda x: x[1], reverse=True)
    return merged[:effective_k], intent

# ── END INTENT-AWARE ADAPTIVE RAG RETRIEVAL ────────────────────────────────

# Load available namespaces helper
def get_available_namespaces(api_key):
    pc_client = get_pinecone_client(api_key)
    if pc_client:
        try:
            index = pc_client.Index(INDEX_NAME)
            stats = index.describe_index_stats()
            return list(stats.get('namespaces', {}).keys())
        except:
            return []
    return []

# Sidebar Dynamic Credentials and Configuration
pinecone_api_key = DEFAULT_PINECONE_API_KEY
gemini_api_key = DEFAULT_GEMINI_API_KEY

if app_mode == "🔍 YCE Virtual RAG Search":
    st.sidebar.markdown("---")
    st.sidebar.markdown("### ⚙️ Pinecone Settings")
    pinecone_api_key = st.sidebar.text_input("Pinecone API Key", value=DEFAULT_PINECONE_API_KEY, type="password")
    
    if not pinecone_api_key:
        st.warning("Please provide a Pinecone API Key to continue.")
        st.stop()
        
    os.environ["PINECONE_API_KEY"] = pinecone_api_key
    pc = get_pinecone_client(pinecone_api_key)
    
    # Ensure 'yce' index exists
    if pc:
        try:
            existing_indexes = [idx.name for idx in pc.list_indexes()]
            if INDEX_NAME not in existing_indexes:
                st.sidebar.info(f"Index '{INDEX_NAME}' not found. Initializing serverless index...")
                pc.create_index(
                    name=INDEX_NAME,
                    dimension=384,
                    metric="cosine",
                    spec=ServerlessSpec(cloud="aws", region="us-east-1")
                )
                st.sidebar.success(f"Index '{INDEX_NAME}' created successfully!")
                st.cache_resource.clear()
            else:
                st.sidebar.success(f"Connected to Pinecone index: **{INDEX_NAME}**")
        except Exception as e:
            st.sidebar.error(f"Failed to check/create index: {str(e)}")
            
    # Index stats dashboard
    if pc:
        try:
            index = pc.Index(INDEX_NAME)
            stats = index.describe_index_stats()
            st.sidebar.markdown("---")
            st.sidebar.markdown("### 📊 Index Statistics")
            st.sidebar.metric("Total Vectors", f"{stats['total_vector_count']:,}")
            
            namespaces = stats.get('namespaces', {})
            if namespaces:
                st.sidebar.markdown("**Namespaces:**")
                for ns, data in namespaces.items():
                    st.sidebar.text(f"• {ns} ({data['vector_count']:,} vectors)")
            else:
                st.sidebar.info("No namespaces uploaded yet.")
        except Exception as e:
            st.sidebar.error(f"Could not load index stats: {str(e)}")

else:  # Gemini Chatbot Settings
    st.sidebar.markdown("---")
    st.sidebar.markdown("### ⚙️ Gemini Settings")
    gemini_api_key = st.sidebar.text_input("Gemini API Key", value=DEFAULT_GEMINI_API_KEY, type="password")
    
    if not gemini_api_key:
        st.warning("Please provide a Gemini API Key to continue.")
        st.stop()
        
    os.environ["GOOGLE_API_KEY"] = gemini_api_key
    
    # Fetch namespaces list for RAG integration
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📚 Chatbot RAG Source")
    
    # Retrieve namespaces using Pinecone API key
    namespaces = get_available_namespaces(DEFAULT_PINECONE_API_KEY)
    
    rag_mode = st.sidebar.selectbox(
        "RAG Document Source",
        options=["None (General Chat)"] + namespaces,
        index=0,
        help="Select a namespace to chat with. The chatbot will answer questions based on the selected document."
    )
    
    top_k_rag = 4
    speed_mode = False
    if rag_mode != "None (General Chat)":
        col1, col2 = st.sidebar.columns([3, 2])
        with col1:
            top_k_rag = st.slider(
                "Chunks (K)",
                min_value=2, max_value=12, value=4,
                help="Number of document chunks to retrieve per query."
            )
        with col2:
            speed_mode = st.checkbox(
                "⚡ Fast Mode",
                value=False,
                help="Skips HyDE (no extra Gemini call). Much faster, slightly less accurate."
            )
        if speed_mode:
            st.sidebar.success("⚡ Fast Mode: parallel retrieval only, no HyDE.")
        else:
            st.sidebar.info("🧠 Smart Mode: HyDE + parallel retrieval (cached).")
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🤖 Model Settings")
    model_option = st.sidebar.selectbox(
        "Select Model",
        options=["gemini-1.5-flash", "gemini-1.5-pro", "gemini-2.0-flash", "Custom Model Name"],
        index=0
    )
    
    if model_option == "Custom Model Name":
        model_name = st.sidebar.text_input("Enter Model String", value="gemini-1.5-flash")
    else:
        model_name = model_option
        
    temperature = st.sidebar.slider("Temperature (Creativity)", min_value=0.0, max_value=2.0, value=0.7, step=0.1)
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🎭 System Personality")
    personality_preset = st.sidebar.selectbox(
        "Choose Preset Persona",
        options=["Helpful AI Assistant", "Expert Python Programmer", "Creative Storyteller", "Sarcastic Tech Sage", "Custom Prompt"],
        index=0
    )
    
    if personality_preset == "Helpful AI Assistant":
        system_prompt = "You are a helpful, extremely polite, and concise AI assistant. You answer questions accurately and format your output beautifully."
    elif personality_preset == "Expert Python Programmer":
        system_prompt = "You are an expert Python software engineer. Explain your code step-by-step, write clean, well-commented code blocks, and follow PEP-8 conventions."
    elif personality_preset == "Creative Storyteller":
        system_prompt = "You are a creative writer and storyteller. Use vivid imagery, suspenseful hooks, and expressive descriptions in all your responses."
    elif personality_preset == "Sarcastic Tech Sage":
        system_prompt = "You are a knowledgeable but highly sarcastic tech sage. You answer correctly but can't help making playful, snarky remarks about user queries."
    else:
        system_prompt = st.sidebar.text_area("System Prompt", value="You are a helpful assistant.")
        
    st.sidebar.markdown("---")
    if st.sidebar.button("🧹 Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ----------------- PAGE ROUTING -----------------

if app_mode == "🔍 YCE Virtual RAG Search":
    # Header Section
    st.markdown("""
    <div class="dashboard-header">
        <h1 class="dashboard-title">🔍 YCE Virtual RAG Search</h1>
        <p class="dashboard-subtitle">Upload PDFs, chunk document text, embed via Sentence Transformers, and retrieve matches from Pinecone.</p>
    </div>
    """, unsafe_allow_html=True)
    
    tab_search, tab_upload = st.tabs(["🔎 Query Database", "📤 Upload & Index Document"])
    
    with tab_upload:
        st.markdown("### 📄 Upload Document to Vector Store")
        uploaded_file = st.file_uploader("Select a PDF file", type=["pdf"])
        
        default_ns = ""
        if uploaded_file:
            clean_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in uploaded_file.name)
            default_ns = clean_name.replace(".pdf", "").strip("_")
            
        namespace = st.text_input("Pinecone Namespace", value=default_ns, help="Documents are isolated by namespaces.")
        
        chunk_size = st.slider("Chunk Size (characters)", min_value=200, max_value=2000, value=1000, step=100)
        chunk_overlap = st.slider("Chunk Overlap (characters)", min_value=0, max_value=500, value=200, step=50)
        
        if st.button("🚀 Process & Upload to Pinecone", use_container_width=True):
            if not uploaded_file:
                st.error("Please upload a PDF file first.")
            elif not namespace.strip():
                st.error("Please specify a namespace.")
            else:
                with st.spinner("Processing document..."):
                    try:
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                            tmp_file.write(uploaded_file.read())
                            tmp_path = tmp_file.name
                        
                        st.text("1. Extracting text from PDF...")
                        loader = PyPDFLoader(tmp_path)
                        documents = loader.load()
                        
                        st.text("2. Chunking document text...")
                        text_splitter = RecursiveCharacterTextSplitter(
                            chunk_size=chunk_size,
                            chunk_overlap=chunk_overlap
                        )
                        chunks = text_splitter.split_documents(documents)
                        
                        # Windows file-lock bug protection
                        try:
                            os.unlink(tmp_path)
                        except Exception:
                            pass
                        
                        st.info(f"Generated {len(chunks)} chunks from {len(documents)} pages.")
                        
                        st.text("3. Generating embeddings & uploading to Pinecone...")
                        vector_store = PineconeVectorStore.from_documents(
                            documents=chunks,
                            embedding=embeddings,
                            index_name=INDEX_NAME,
                            namespace=namespace,
                            pinecone_api_key=pinecone_api_key
                        )
                        
                        st.success(f"Successfully uploaded {len(chunks)} vectors to namespace '{namespace}'!")
                        st.cache_resource.clear()
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f"An error occurred during processing: {str(e)}")
                        
    with tab_search:
        st.markdown("### 🔍 Search Vectors")
        
        available_namespaces = []
        try:
            index = pc.Index(INDEX_NAME)
            stats = index.describe_index_stats()
            available_namespaces = list(stats.get('namespaces', {}).keys())
        except:
            pass
            
        search_ns = st.selectbox(
            "Select Namespace to query",
            options=available_namespaces if available_namespaces else ["default"],
            index=0 if available_namespaces else None,
            help="Select which document namespace to search within."
        )
        
        query = st.text_input("Enter your query:", placeholder="e.g. What is the total revenue?")
        top_k = st.slider("Number of results to retrieve (K)", min_value=1, max_value=10, value=4)
        
        if st.button("🔍 Search", use_container_width=True) or query:
            if not query.strip():
                if query:
                    st.warning("Please enter a non-empty search query.")
            else:
                with st.spinner("Searching Pinecone database..."):
                    try:
                        vector_store = PineconeVectorStore(
                            index_name=INDEX_NAME,
                            embedding=embeddings,
                            pinecone_api_key=pinecone_api_key,
                            namespace=search_ns
                        )
                        
                        results = vector_store.similarity_search_with_score(query, k=top_k)
                        
                        if not results:
                            st.info("No matching results found in this namespace.")
                        else:
                            st.markdown(f"#### 🎯 Retained Matches ({len(results)} matches retrieved):")
                            for idx, (doc, score) in enumerate(results):
                                source = doc.metadata.get("source", "Unknown Source")
                                filename = os.path.basename(source)
                                page = doc.metadata.get("page", 0) + 1
                                
                                st.markdown(f"""
                                <div class="result-card">
                                    <div class="card-meta">
                                        <span class="card-filename">📁 {filename}</span>
                                        <div>
                                            <span class="meta-badge-score">Similarity: {score:.4f}</span>
                                            <span class="meta-badge-page">Page {page}</span>
                                        </div>
                                    </div>
                                    <div class="card-text">{doc.page_content}</div>
                                </div>
                                """, unsafe_allow_html=True)
                    except Exception as e:
                        st.error(f"Search failed: {str(e)}")

else:  # Gemini Chatbot Mode
    # Header Section
    is_rag_active = (rag_mode != "None (General Chat)")
    subtitle_desc = f"Grounded QA mode active on document <b>{rag_mode}</b>." if is_rag_active else "General reasoning mode active."
    
    st.markdown(f"""
    <div class="dashboard-header">
        <h1 class="dashboard-title">💬 Gemini RAG Chatbot</h1>
        <p class="dashboard-subtitle">Powered by <b>{model_name}</b> and LangChain. {subtitle_desc}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Render Chat History
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
            if "sources" in message and message["sources"]:
                with st.expander("📂 View Retrieved Sources"):
                    for src in message["sources"]:
                        st.markdown(f"""
                        **File:** `{src['filename']}` | **Page:** `{src['page']}` | **Score:** `{src['score']:.4f}`
                        ```text
                        {src['text']}
                        ```
                        """)
            
    # Chat Input & Streaming Logic
    if prompt := st.chat_input("Ask me anything..."):
        with st.chat_message("user"):
            st.write(prompt)
            
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Build prompt & retrieve chunks if RAG is active
        context_text = ""
        sources = []
        map_reduce_summary = ""   # set when summarize_full uses map-reduce path

        if is_rag_active:
            intent_label = "general"
            with st.spinner("🧠 Intent-aware retrieval in progress..."):
                try:
                    vector_store = PineconeVectorStore(
                        index_name=INDEX_NAME,
                        embedding=embeddings,
                        pinecone_api_key=DEFAULT_PINECONE_API_KEY,
                        namespace=rag_mode
                    )
                    retrieved_chunks, intent_label = adaptive_retrieve(
                        vector_store,
                        prompt,
                        k=top_k_rag,
                        gemini_api_key=gemini_api_key,
                        model_name=model_name,
                        namespace=rag_mode,
                        speed_mode=speed_mode
                    )

                    # -- FULL DOCUMENT SUMMARY: Map-Reduce path --
                    if intent_label == "summarize_full":
                        st.info(f"📚 Retrieved {len(retrieved_chunks)} chunks from across the document. Generating summary…")
                        domain = _detect_document_domain(rag_mode)
                        doc_desc = _DOMAIN_DESCRIPTIONS.get(domain, "document")
                        with st.spinner("⚡ Building document summary (single Gemini call)..."):
                            map_reduce_summary = _fast_summarize(
                                retrieved_chunks, gemini_api_key, model_name, doc_desc
                            )
                        # Build light sources list for display
                        for doc, score in retrieved_chunks:
                            source_path = doc.metadata.get("source", "Unknown Source")
                            sources.append({
                                "filename": os.path.basename(source_path),
                                "page": doc.metadata.get("page", 0) + 1,
                                "score": score,
                                "text": doc.page_content[:300] + "..."
                            })

                    else:
                        LOW_SCORE_THRESHOLD = 0.25
                        if retrieved_chunks and retrieved_chunks[0][1] < LOW_SCORE_THRESHOLD:
                            st.warning(
                                f"⚠️ Best similarity score is only {retrieved_chunks[0][1]:.3f}. "
                                "This topic may not be well-covered in the document."
                            )

                        # Cap total context at ~6000 chars to avoid token overflows
                        MAX_CONTEXT_CHARS = 6000
                        chars_used = 0
                        context_blocks = []
                        for doc, score in retrieved_chunks:
                            chunk_text = doc.page_content
                            if chars_used + len(chunk_text) > MAX_CONTEXT_CHARS:
                                chunk_text = chunk_text[:MAX_CONTEXT_CHARS - chars_used]
                                if chunk_text:
                                    source_path = doc.metadata.get("source", "Unknown")
                                    filename = os.path.basename(source_path)
                                    page = doc.metadata.get("page", 0) + 1
                                    context_blocks.append(
                                        f"--- {filename}, P{page}, score={score:.2f} ---\n{chunk_text}[truncated]"
                                    )
                                break
                            source_path = doc.metadata.get("source", "Unknown")
                            filename = os.path.basename(source_path)
                            page = doc.metadata.get("page", 0) + 1
                            context_blocks.append(
                                f"--- {filename}, P{page}, score={score:.2f} ---\n{chunk_text}"
                            )
                            chars_used += len(chunk_text)
                            sources.append({
                                "filename": filename,
                                "page": page,
                                "score": score,
                                "text": chunk_text
                            })
                        context_text = "\n\n".join(context_blocks)

                except Exception as e:
                    intent_label = "general"
                    err_msg = str(e)
                    if "429" in err_msg or "quota" in err_msg.lower() or "rate" in err_msg.lower():
                        st.error(
                            "🚫 **Gemini API rate limit reached.**\n\n"
                            "The free tier allows ~15 requests/minute. Please wait 60 seconds and try again, "
                            "or enable **⚡ Fast Mode** in the sidebar to reduce API calls."
                        )
                    else:
                        st.error(f"Retrieval failed: {err_msg}")

        # -- If map-reduce summary was produced, display it directly and skip LLM call --
        if map_reduce_summary:
            with st.chat_message("assistant"):
                st.markdown(map_reduce_summary)
                if sources:
                    with st.expander(f"📂 Sources used ({len(sources)} chunks from document)"):
                        for src in sources:
                            st.markdown(f"`{src['filename']}` | Page {src['page']}")
            st.session_state.messages.append({
                "role": "assistant",
                "content": map_reduce_summary,
                "sources": sources
            })
        else:
            # -- Standard path: build system prompt + single Gemini call --
            if is_rag_active:
                if intent_label == "toc":
                    task_instructions = """
- The user wants the complete Table of Contents / list of all sections.
- Scan EVERY context chunk for section headings, numbered items, or topic titles.
- Output a COMPLETE numbered outline of ALL sections and subsections you find.
- Do NOT ask the user which section they want — just output the full list immediately.
- Do NOT offer to generate summaries — just list all the sections found.
- Format: numbered list with sub-items indented."""
                elif intent_label == "summarize_section":
                    task_instructions = """
- The user wants a summary of a SPECIFIC section.
- Gather ALL relevant passages about that section from the context chunks.
- Write a comprehensive summary: purpose, key data/figures, findings, conclusions.
- Use bullet points and clear headings. Include exact figures and dates as they appear.
- Do NOT ask follow-up questions. Just write the summary with what is available."""
                else:
                    task_instructions = """
- Read ALL context passages carefully before answering.
- Synthesize across MULTIPLE passages — the answer may be split across chunks.
- Quote exact figures, amounts, and values as they appear.
- Only say 'I cannot find the answer' if NO chunk contains relevant data.
- Do NOT invent facts."""

                custom_system_prompt = (
                    f"You are a highly accurate document analyst.\n\n"
                    f"Instructions:{task_instructions}\n\n"
                    f"Context from document ({len(sources)} chunks, ~{len(context_text)} chars):\n"
                    f"{context_text}\n\n"
                    f"Personality: {system_prompt}"
                )
            else:
                custom_system_prompt = system_prompt

            langchain_messages = [SystemMessage(content=custom_system_prompt)]
            for msg in st.session_state.messages:
                if msg["role"] == "user":
                    langchain_messages.append(HumanMessage(content=msg["content"]))
                elif msg["role"] == "assistant":
                    langchain_messages.append(AIMessage(content=msg["content"]))

            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                full_response = ""
                try:
                    temp_val = temperature if not is_rag_active else min(temperature, 0.4)
                    stream, model_used = safe_llm_stream(
                        selected_model=model_name,
                        api_key=gemini_api_key,
                        messages=langchain_messages,
                        temperature=temp_val
                    )
                    
                    # Highlight if we had to fall back to a stable model due to 503/429
                    if model_used != model_name:
                        st.info(f"ℹ️ Switched automatically to `{model_used}` to bypass traffic spikes.")
                        
                    for chunk in stream:
                        full_response += chunk.content
                        message_placeholder.markdown(full_response + "▮")
                    message_placeholder.markdown(full_response)

                    if sources:
                        with st.expander("📂 View Retrieved Sources"):
                            for src in sources:
                                st.markdown(
                                    f"**File:** `{src['filename']}` | "
                                    f"**Page:** `{src['page']}` | "
                                    f"**Score:** `{src['score']:.4f}`\n"
                                    f"```text\n{src['text']}\n```"
                                )

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": full_response,
                        "sources": sources
                    })

                except Exception as e:
                    err_msg = str(e)
                    if "429" in err_msg or "quota" in err_msg.lower() or "rate" in err_msg.lower():
                        message_placeholder.error(
                            "🚫 **Gemini API rate limit reached.** "
                            "Please wait ~60 seconds and try again, or enable **⚡ Fast Mode** in the sidebar."
                        )
                    else:
                        message_placeholder.error(f"Error calling Gemini API: {err_msg}")
                        st.info("Tip: Check your API key and model name in the sidebar.")

