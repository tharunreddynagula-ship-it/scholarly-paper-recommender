import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(
    page_title="Scholarly Paper Recommender",
    page_icon="📚",
    layout="wide"
)

st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #0f172a 0%, #111827 45%, #1e293b 100%);
    color: white;
}

h1, h2, h3 {
    color: #f8fafc;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #020617 0%, #111827 100%);
}

.stButton > button {
    background: linear-gradient(90deg, #2563eb, #7c3aed);
    color: white;
    border-radius: 12px;
    border: none;
    padding: 0.6rem 1.2rem;
    font-weight: bold;
}

.stButton > button:hover {
    background: linear-gradient(90deg, #1d4ed8, #6d28d9);
    color: white;
}

[data-testid="stMetric"] {
    background: rgba(255, 255, 255, 0.08);
    padding: 15px;
    border-radius: 15px;
    border: 1px solid rgba(255,255,255,0.15);
}

div[data-testid="stAlert"] {
    border-radius: 15px;
}

.block-container {
    padding-top: 2rem;
}
</style>
""", unsafe_allow_html=True)

st.title("📚 Scholarly Paper Recommendation System")

st.success(
    "Scholarly Paper Recommendation System using OpenAlex Metadata and Text Similarity"
)

st.markdown("""
This app recommends scholarly papers using **text similarity** from paper titles, abstracts, and topics.

Dataset source: **OpenAlex** | Corpus size: **1000 Machine Learning papers**
""")

@st.cache_data
def load_data():
    return pd.read_csv("papers_1000_final.csv")

df = load_data()

df["combined_text"] = (
    df["title"].fillna("") + " " +
    df["abstract"].fillna("") + " " +
    df["topics"].fillna("")
)

@st.cache_resource
def build_similarity(data):
    tfidf = TfidfVectorizer(
        stop_words="english",
        max_features=10000
    )

    tfidf_matrix = tfidf.fit_transform(data["combined_text"])
    similarity_matrix = cosine_similarity(tfidf_matrix)

    return similarity_matrix

def clean_value(value):
    if pd.isna(value):
        return "Not Available"
    return value

similarity_matrix = build_similarity(df)

# Sidebar
st.sidebar.header("📊 Dataset Overview")
st.sidebar.metric("Total Papers", len(df))
st.sidebar.metric(
    "Year Range",
    f"{int(df['year'].min())} - {int(df['year'].max())}"
)
st.sidebar.metric(
    "Average Citations",
    int(df["citation_count"].mean())
)

# Paper Selection
st.subheader("🔎 Search and Select a Research Paper")

search_query = st.text_input(
    "Type a keyword from the paper title:",
    placeholder="Example: federated, healthcare, privacy, deep learning"
)

paper_titles = sorted(df["title"].dropna().unique())

if search_query:
    filtered_titles = [
        title for title in paper_titles
        if search_query.lower() in title.lower()
    ]
else:
    filtered_titles = paper_titles

if len(filtered_titles) == 0:
    st.warning("No papers found. Try another keyword.")
    st.stop()

paper_title = st.selectbox(
    "Choose one paper:",
    filtered_titles
)

top_n = st.slider(
    "Number of recommendations",
    5,
    20,
    10
)

if st.button("🚀 Get Recommendations"):

    idx = df[df["title"] == paper_title].index[0]
    selected_paper = df.iloc[idx]

    # Selected Paper
    st.markdown("## 📌 Selected Paper")

    st.info(f"**{selected_paper['title']}**")

    st.write(f"**Year:** {clean_value(selected_paper.get('year'))}")
    st.write(f"**Authors:** {clean_value(selected_paper.get('authors'))}")
    st.write(f"**Citations:** {clean_value(selected_paper.get('citation_count'))}")
    st.write(f"**Topics:** {clean_value(selected_paper.get('topics'))}")

    selected_paper_url = selected_paper.get("paper_id")

    if isinstance(selected_paper_url, str) and selected_paper_url.startswith("http"):
        st.link_button(
            "🔗 Open Selected Paper",
            selected_paper_url
        )

    selected_abstract = str(
        selected_paper.get("abstract", "")
    )

    if selected_abstract and selected_abstract != "nan":
        st.write("**Abstract:**")
        st.write(selected_abstract[:800] + "...")

    # Recommendations
    scores = list(enumerate(similarity_matrix[idx]))

    scores = sorted(
        scores,
        key=lambda x: x[1],
        reverse=True
    )

    scores = [
        (i, score)
        for i, score in scores
        if i != idx
    ][:top_n]

    st.markdown("## ✅ Recommended Papers")

    for rank, (paper_idx, score) in enumerate(scores, start=1):

        paper = df.iloc[paper_idx]

        with st.container():

            st.markdown(
                f"### {rank}. {paper['title']}"
            )

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(
                    "Similarity",
                    f"{score:.3f}"
                )

            with col2:
                st.metric(
                    "Year",
                    int(paper["year"])
                )

            with col3:
                st.metric(
                    "Citations",
                    int(paper["citation_count"])
                )

            st.write(
                f"**Authors:** {clean_value(paper.get('authors'))}"
            )

            st.write(
                f"**Topics:** {clean_value(paper.get('topics'))}"
            )

            paper_url = paper.get("paper_id")

            if isinstance(paper_url, str) and paper_url.startswith("http"):
                st.link_button(
                    "🔗 Open Paper",
                    paper_url
                )

            abstract = str(
                paper.get("abstract", "")
            )

            if abstract and abstract != "nan":
                st.write("**Abstract Preview:**")
                st.write(
                    abstract[:500] + "..."
                )

            st.divider()