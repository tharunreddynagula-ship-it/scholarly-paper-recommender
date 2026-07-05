import streamlit as st
import pandas as pd
from html import escape
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="ScholarMatch", page_icon="🎓", layout="wide", initial_sidebar_state="collapsed")

st.markdown(r"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');
*{font-family:'Inter',sans-serif}.stApp{background:#f7f8fa;color:#111827}.block-container{padding:2rem 3rem 3rem 3rem;max-width:1300px}#MainMenu,footer,header{visibility:hidden}.topbar{background:#fff;border:1px solid #e5e7eb;border-radius:22px;padding:18px 24px;box-shadow:0 12px 30px rgba(15,23,42,.06);display:flex;align-items:center;justify-content:space-between;margin-bottom:24px}.brand{font-size:24px;font-weight:900;color:#111827}.nav-pill{display:inline-block;background:#f3f4f6;color:#374151;border:1px solid #e5e7eb;padding:8px 14px;border-radius:999px;font-size:13px;font-weight:800;margin-left:8px}.hero{background:linear-gradient(135deg,#fff 0%,#f8fbff 100%);border:1px solid #e5e7eb;border-radius:28px;padding:42px;box-shadow:0 18px 45px rgba(15,23,42,.08);margin-bottom:24px}.badge{display:inline-block;background:#eff6ff;color:#1d4ed8;border:1px solid #bfdbfe;padding:7px 14px;border-radius:999px;font-size:13px;font-weight:800;margin-bottom:15px}.hero-title{font-size:48px;font-weight:900;letter-spacing:-1.5px;color:#111827;margin-bottom:14px;line-height:1.05}.hero-text{font-size:17px;color:#4b5563;line-height:1.7;max-width:850px}.card{background:#fff;border:1px solid #e5e7eb;border-radius:24px;padding:26px;box-shadow:0 14px 32px rgba(15,23,42,.06);transition:all .25s ease;margin-bottom:20px}.card:hover{transform:translateY(-4px);box-shadow:0 24px 45px rgba(15,23,42,.11);border-color:#bfdbfe}.kpi-icon{width:50px;height:50px;background:#eff6ff;color:#2563eb;border-radius:16px;display:flex;align-items:center;justify-content:center;font-size:24px;margin-bottom:14px}.kpi-label{color:#6b7280;font-size:13px;font-weight:800;text-transform:uppercase}.kpi-value{color:#111827;font-size:30px;font-weight:900;margin-top:4px}.kpi-sub{color:#6b7280;font-size:13px;margin-top:4px}.section-title{font-size:26px;font-weight:900;color:#111827;margin-bottom:6px}.section-subtitle{color:#6b7280;font-size:15px;line-height:1.6}.line{width:46px;height:4px;background:#2563eb;border-radius:999px;margin:14px 0 22px 0}.workflow{display:flex;flex-wrap:wrap;gap:12px;margin-top:18px}.workflow-step{background:#f9fafb;border:1px solid #e5e7eb;color:#374151;padding:14px 16px;border-radius:16px;font-weight:800}.start-box{text-align:center;padding:44px}.start-icon{font-size:60px;margin-bottom:10px}.stButton>button{background:linear-gradient(90deg,#2563eb,#1d4ed8);color:#fff;border:none;border-radius:14px;padding:.8rem 1.6rem;font-weight:900;box-shadow:0 12px 22px rgba(37,99,235,.25);transition:all .25s ease}.stButton>button:hover{transform:translateY(-2px) scale(1.01);background:linear-gradient(90deg,#1d4ed8,#1e40af);color:#fff}.stTextInput input{background:#fff!important;border:1px solid #d1d5db!important;border-radius:14px!important;padding:14px!important;color:#111827!important}.stSelectbox div[data-baseweb="select"]{background:#fff!important;border-radius:14px!important;border:1px solid #d1d5db!important}.paper-card{background:#fff;border:1px solid #e5e7eb;border-radius:24px;padding:25px;box-shadow:0 16px 36px rgba(15,23,42,.06);margin-bottom:18px;transition:all .25s ease}.paper-card:hover{transform:translateY(-4px);box-shadow:0 24px 45px rgba(15,23,42,.11);border-color:#bfdbfe}.selected-card{background:linear-gradient(135deg,#fff,#eff6ff);border:1px solid #bfdbfe;border-radius:24px;padding:25px;box-shadow:0 18px 42px rgba(37,99,235,.10);margin-bottom:20px}.paper-title{font-size:21px;font-weight:900;color:#111827;line-height:1.35;margin-bottom:14px}.small-badge{display:inline-block;background:#f3f4f6;color:#374151;border:1px solid #e5e7eb;padding:6px 12px;border-radius:999px;font-size:13px;font-weight:800;margin-right:7px;margin-bottom:8px}.score-badge{display:inline-block;background:#2563eb;color:#fff;padding:6px 12px;border-radius:999px;font-size:13px;font-weight:900;margin-right:7px;margin-bottom:8px}.meta{color:#4b5563;font-size:14px;line-height:1.65;margin-top:12px}.abstract-box{background:#f9fafb;border-left:4px solid #2563eb;border-radius:14px;padding:16px;color:#374151;line-height:1.6;margin-bottom:16px}.empty-state{background:#fff;border:1px dashed #cbd5e1;border-radius:26px;padding:48px;text-align:center;color:#6b7280;box-shadow:0 16px 36px rgba(15,23,42,.05)}.doc-block{background:#fff;border:1px solid #e5e7eb;border-radius:22px;padding:24px;box-shadow:0 14px 32px rgba(15,23,42,.06);margin-bottom:18px}.doc-title{font-size:20px;font-weight:900;color:#111827;margin-bottom:8px}.doc-text{color:#4b5563;line-height:1.75;font-size:15px}.footer{text-align:center;color:#9ca3af;font-size:12px;margin-top:30px}
</style>
""", unsafe_allow_html=True)

def safe_text(value):
    if pd.isna(value): return "Not Available"
    return escape(str(value))

def safe_int(value):
    try:
        if pd.isna(value): return 0
        return int(value)
    except Exception:
        return 0

def short_text(text, length=600):
    if text is None or pd.isna(text): return ""
    text = str(text)
    if text.lower() == "nan" or text.strip() == "": return ""
    return text[:length] + ("..." if len(text) > length else "")

@st.cache_data
def load_data():
    return pd.read_csv("papers_1000_final.csv")

df = load_data()
df["combined_text"] = df["title"].fillna("") + " " + df["abstract"].fillna("") + " " + df["topics"].fillna("")

@st.cache_resource
def build_similarity(data):
    tfidf = TfidfVectorizer(stop_words="english", max_features=10000)
    tfidf_matrix = tfidf.fit_transform(data["combined_text"])
    return cosine_similarity(tfidf_matrix)

similarity_matrix = build_similarity(df)

if "page" not in st.session_state: st.session_state.page = "dashboard"
if "recent_searches" not in st.session_state: st.session_state.recent_searches = []

st.markdown("""
<div class="topbar"><div class="brand">🎓 ScholarMatch</div><div><span class="nav-pill">Machine Learning</span><span class="nav-pill">OpenAlex</span><span class="nav-pill">TF-IDF</span></div></div>
""", unsafe_allow_html=True)

nav1, nav2, nav3 = st.columns([1,1,5])
with nav1:
    if st.button("🏠 Dashboard"):
        st.session_state.page = "dashboard"; st.rerun()
with nav2:
    if st.button("🧠 About Model"):
        st.session_state.page = "about"; st.rerun()

if st.session_state.page == "dashboard":
    st.markdown("""<div class="hero"><div class="badge">Machine Learning Project · OpenAlex Dataset</div><div class="hero-title">Scholarly Paper Recommendation System</div><div class="hero-text">A professional research discovery tool that recommends similar scholarly papers using OpenAlex metadata, TF-IDF text similarity, and cosine similarity.</div></div>""", unsafe_allow_html=True)
    c1,c2,c3,c4 = st.columns(4)
    with c1:
        st.markdown(f"""<div class="card"><div class="kpi-icon">📄</div><div class="kpi-label">Total Papers</div><div class="kpi-value">{len(df):,}</div><div class="kpi-sub">Machine Learning Papers</div></div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class="card"><div class="kpi-icon">📅</div><div class="kpi-label">Year Range</div><div class="kpi-value">{safe_int(df['year'].min())} - {safe_int(df['year'].max())}</div><div class="kpi-sub">Publication Years</div></div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div class="card"><div class="kpi-icon">📈</div><div class="kpi-label">Average Citations</div><div class="kpi-value">{safe_int(df['citation_count'].mean())}</div><div class="kpi-sub">Per Paper</div></div>""", unsafe_allow_html=True)
    with c4:
        st.markdown("""<div class="card"><div class="kpi-icon">🗄️</div><div class="kpi-label">Data Source</div><div class="kpi-value">OpenAlex</div><div class="kpi-sub">Scholarly Database</div></div>""", unsafe_allow_html=True)
    st.markdown("""<div class="card"><div class="section-title">System Workflow</div><div class="line"></div><div class="section-subtitle">The project transforms scholarly metadata into numerical vectors and ranks papers by similarity.</div><div class="workflow"><div class="workflow-step">OpenAlex Dataset</div><div class="workflow-step">Text Preprocessing</div><div class="workflow-step">TF-IDF Vectorization</div><div class="workflow-step">Cosine Similarity</div><div class="workflow-step">Top-N Recommendations</div></div></div>""", unsafe_allow_html=True)
    left,right = st.columns([1.25,1])
    with left:
        st.markdown("""<div class="card start-box"><div class="start-icon">🚀📚</div><h2>Start Paper Recommendation</h2><p style="color:#6b7280;">Open the recommendation page to search research papers and generate similar paper suggestions.</p></div>""", unsafe_allow_html=True)
        if st.button("🚀 Start Now"):
            st.session_state.page = "search"; st.rerun()
    with right:
        recent = st.session_state.recent_searches[-5:]
        items = "<br>".join([f"✔ {escape(x)}" for x in reversed(recent)]) if recent else "No searches yet."
        st.markdown(f"""<div class="card"><div class="section-title">Recent Searches</div><div class="line"></div><div class="section-subtitle">{items}</div></div>""", unsafe_allow_html=True)

elif st.session_state.page == "search":
    if st.button("⬅ Back to Dashboard"):
        st.session_state.page = "dashboard"; st.rerun()
    st.markdown("""<div class="hero"><div class="badge">Paper Search</div><div class="hero-title">Find Similar Research Papers</div><div class="hero-text">Search for a paper title, choose one paper, and generate ranked recommendations based on text similarity.</div></div>""", unsafe_allow_html=True)
    st.markdown("""<div class="card"><div class="section-title">Search and Select a Research Paper</div><div class="line"></div><div class="section-subtitle">Choose a paper from the dataset and generate Top-N similar recommendations.</div></div>""", unsafe_allow_html=True)
    left,right = st.columns([1.45,.75])
    with left:
        search_query = st.text_input("Search keyword", placeholder="Example: federated, healthcare, privacy, deep learning")
        paper_titles = sorted(df["title"].dropna().unique())
        filtered_titles = [title for title in paper_titles if search_query.lower() in title.lower()] if search_query else paper_titles
        if len(filtered_titles) == 0:
            st.warning("No papers found. Try another keyword."); st.stop()
        paper_title = st.selectbox("Select paper", filtered_titles)
    with right:
        top_n = st.slider("Number of recommendations", 5, 20, 10)
        get_recommendations = st.button("Generate Recommendations")
    if not get_recommendations:
        st.markdown("""<div class="empty-state"><div style="font-size:58px;">🔍📑</div><h2>Ready to Discover Papers</h2><p>Search and select a paper to begin generating recommendations.</p></div>""", unsafe_allow_html=True)
    if get_recommendations:
        idx = df[df["title"] == paper_title].index[0]
        selected_paper = df.iloc[idx]
        if paper_title not in st.session_state.recent_searches: st.session_state.recent_searches.append(paper_title)
        st.markdown("## Selected Paper")
        st.markdown(f"""<div class="selected-card"><div class="paper-title">{safe_text(selected_paper.get('title'))}</div><span class="small-badge">Year: {safe_text(selected_paper.get('year'))}</span><span class="small-badge">Citations: {safe_text(selected_paper.get('citation_count'))}</span><span class="small-badge">OpenAlex</span><div class="meta"><b>Authors:</b> {safe_text(selected_paper.get('authors'))}<br><b>Topics:</b> {safe_text(selected_paper.get('topics'))}</div></div>""", unsafe_allow_html=True)
        selected_url = selected_paper.get("paper_id")
        if isinstance(selected_url, str) and selected_url.startswith("http"): st.link_button("Open Selected Paper", selected_url)
        abstract = short_text(selected_paper.get("abstract"), 800)
        if abstract: st.markdown(f"""<div class="abstract-box"><b>Abstract Preview</b><br><br>{escape(abstract)}</div>""", unsafe_allow_html=True)
        scores = list(enumerate(similarity_matrix[idx])); scores = sorted(scores, key=lambda x:x[1], reverse=True); scores = [(i,score) for i,score in scores if i != idx][:top_n]
        st.markdown("## Recommended Papers")
        for rank,(paper_idx,score) in enumerate(scores, start=1):
            paper = df.iloc[paper_idx]
            st.markdown(f"""<div class="paper-card"><div class="paper-title">{rank}. {safe_text(paper.get('title'))}</div><span class="score-badge">Similarity: {score:.3f}</span><span class="small-badge">Year: {safe_text(paper.get('year'))}</span><span class="small-badge">Citations: {safe_text(paper.get('citation_count'))}</span><div class="meta"><b>Authors:</b> {safe_text(paper.get('authors'))}<br><b>Topics:</b> {safe_text(paper.get('topics'))}</div></div>""", unsafe_allow_html=True)
            paper_url = paper.get("paper_id")
            if isinstance(paper_url, str) and paper_url.startswith("http"): st.link_button("Open Paper", paper_url)
            abstract = short_text(paper.get("abstract"), 500)
            if abstract: st.markdown(f"""<div class="abstract-box"><b>Abstract Preview</b><br><br>{escape(abstract)}</div>""", unsafe_allow_html=True)

elif st.session_state.page == "about":
    st.markdown("""<div class="hero"><div class="badge">Model Overview</div><div class="hero-title">How the Recommendation System Works</div><div class="hero-text">The system recommends scholarly papers by comparing the textual content of research papers. It combines each paper title, abstract, and topic information into one text representation, converts this text into numerical features using TF-IDF, and calculates similarity using cosine similarity.</div></div>""", unsafe_allow_html=True)
    st.markdown("""
    <div class="doc-block"><div class="doc-title">1. Dataset</div><div class="doc-text">The dataset contains 1000 machine learning papers collected from OpenAlex. Each record includes title, abstract, authors, topics, year, citation count, and paper identifier.</div></div>
    <div class="doc-block"><div class="doc-title">2. Text Representation</div><div class="doc-text">The title, abstract, and topics are merged into a single text field. This gives the model enough information to compare the content of different research papers.</div></div>
    <div class="doc-block"><div class="doc-title">3. TF-IDF Vectorization</div><div class="doc-text">TF-IDF converts text into numerical vectors. It gives higher importance to meaningful words and lower importance to very common words.</div></div>
    <div class="doc-block"><div class="doc-title">4. Cosine Similarity</div><div class="doc-text">Cosine similarity compares the TF-IDF vectors of papers. A higher similarity score means the papers are more related in textual content.</div></div>
    <div class="doc-block"><div class="doc-title">5. Recommendation Output</div><div class="doc-text">After calculating similarity scores, the system ranks all papers and returns the Top-N most similar papers as recommendations.</div></div>
    <div class="doc-block"><div class="doc-title">Future Improvements</div><div class="doc-text">The system can be improved by using larger datasets, BERT or Sentence Transformer embeddings, citation graph similarity, personalized recommendations, and quantitative metrics such as Precision@K.</div></div>
    """, unsafe_allow_html=True)

st.markdown("""<div class="footer">ScholarMatch · Machine Learning Project · Python · Streamlit · TF-IDF · OpenAlex</div>""", unsafe_allow_html=True)
