# ScholarMatch — Scholarly Paper Recommendation System

ScholarMatch is a machine learning-based research paper recommendation system that helps students and researchers discover related academic papers.

The application uses scholarly metadata from the OpenAlex dataset and recommends similar research papers using **TF-IDF vectorization** and **cosine similarity**.

---

## Project Objective

The main objective of this project is to reduce the time required to search for relevant academic papers.

Users can:

- Search for a research paper by title or keyword
- Select a paper from the dataset
- Generate similar paper recommendations
- Choose the number of recommendations
- View paper titles, authors, topics, publication years and citation counts
- Open the original research paper through its available link

---

## Problem Statement

The number of academic research papers is increasing rapidly. Finding papers that are relevant to a specific research topic can be time-consuming.

Traditional keyword searches may return many unrelated results. ScholarMatch addresses this problem by comparing the textual content of papers and recommending papers with similar titles, abstracts and topics.

---

## Dataset

The project uses scholarly paper metadata obtained from **OpenAlex**, an open scholarly database.

The dataset contains approximately 1,000 research-paper records.

Important fields include:

- Paper title
- Abstract
- Authors
- Topics
- Publication year
- Citation count
- Paper identifier or URL

Dataset file:

```text
papers_1000_final.csv
```

---

## Machine Learning Methodology

### 1. Data Loading

The research-paper dataset is loaded using Pandas.

```python
df = pd.read_csv("papers_1000_final.csv")
```

### 2. Missing-Value Handling

Missing values in the title, abstract and topics fields are replaced with empty text.

### 3. Feature Combination

The following text fields are combined:

```text
Title + Abstract + Topics
```

This combined text represents the content of each research paper.

### 4. TF-IDF Vectorization

TF-IDF stands for **Term Frequency–Inverse Document Frequency**.

It converts research-paper text into numerical vectors.

TF-IDF gives greater importance to terms that:

- Appear frequently in a particular paper
- Appear less frequently across the complete collection of papers

The project uses:

```python
TfidfVectorizer(
    stop_words="english",
    max_features=10000
)
```

### 5. Cosine Similarity

Cosine similarity measures the similarity between two document vectors by calculating the angle between them.

The similarity score generally ranges from:

- `0` — no meaningful similarity
- `1` — very high similarity

The papers are ranked according to their cosine-similarity scores.

### 6. Top-N Recommendations

After the user selects a paper, the system:

1. Finds the selected paper in the dataset
2. Retrieves its similarity scores
3. Removes the selected paper from the results
4. Sorts the remaining papers by similarity
5. Displays the Top-N recommendations

---

## Recommendation Workflow

```text
OpenAlex Dataset
        ↓
Data Cleaning
        ↓
Combine Title, Abstract and Topics
        ↓
TF-IDF Vectorization
        ↓
Cosine Similarity Calculation
        ↓
Rank Similar Papers
        ↓
Top-N Paper Recommendations
```

---

## Technologies Used

- Python
- Streamlit
- Pandas
- NumPy
- Scikit-learn
- TF-IDF Vectorizer
- Cosine Similarity
- OpenAlex Dataset
- GitHub
- Streamlit Community Cloud

---

## Project Structure

```text
scholarly-paper-recommender/
│
├── app.py
├── papers_1000_final.csv
├── model_comparison.csv
├── requirements.txt
└── README.md
```

### File Descriptions

| File | Description |
|---|---|
| `app.py` | Streamlit interface and recommendation-system implementation |
| `papers_1000_final.csv` | Scholarly paper dataset |
| `model_comparison.csv` | Comparison of recommendation approaches |
| `requirements.txt` | Required Python packages |
| `README.md` | Project documentation |

---

## Application Features

### Dashboard

The main dashboard displays:

- Total number of papers
- Publication-year range
- Average citation count
- Dataset source
- Machine learning workflow
- Recent paper searches

### Paper Search

Users can:

- Enter a search keyword
- Filter available paper titles
- Select a research paper
- Choose between 5 and 20 recommendations

### Recommendation Results

For each recommended paper, the system can display:

- Recommendation rank
- Paper title
- Similarity score
- Publication year
- Citation count
- Authors
- Topics
- Abstract preview
- Available paper link

### About Model

The application also explains:

- The project objective
- TF-IDF
- Cosine similarity
- Recommendation workflow
- Model limitations

---

## Model Comparison

The project considered three possible recommendation approaches:

| Model | Information Used | Strength | Limitation |
|---|---|---|---|
| Text Similarity | Title, abstract and topics | Identifies papers with similar words and topics | May not capture citation relationships |
| Graph Similarity | Shared references and citation links | Identifies academically connected papers | May be weak when citation information is limited |
| Hybrid Similarity | Text and citation relationships | Can provide balanced recommendations | Requires more data and computation |

The current deployed application uses **TF-IDF text similarity with cosine similarity**.

---

## Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/tharunreddynagula-ship-it/scholarly-paper-recommender.git
```

### 2. Open the Project Folder

```bash
cd scholarly-paper-recommender
```

### 3. Create a Virtual Environment

Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

macOS or Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 4. Install the Required Libraries

```bash
pip install -r requirements.txt
```

### 5. Run the Streamlit Application

```bash
streamlit run app.py
```

The application should open automatically in the browser.

---

## Requirements

The project uses the following Python libraries:

```text
streamlit
pandas
scikit-learn
numpy
```

---

## Key Findings

- TF-IDF successfully converts unstructured academic text into numerical features.
- Cosine similarity can identify papers with related titles, abstracts and topics.
- Combining multiple text fields provides more information than using only paper titles.
- The system can provide recommendations without labelled training data.
- Recommendation quality depends on the completeness of abstracts and topics.
- Similarity-based recommendations are fast and explainable.

---

## Evaluation

The system was evaluated by reviewing:

- Similarity scores
- Relevance of recommended paper titles
- Similarity of topics and abstracts
- Consistency of recommendations for different selected papers
- Differences between text, graph and hybrid recommendation approaches

Because the dataset does not contain labelled relevance judgments, the current evaluation is mainly qualitative.

Possible future evaluation metrics include:

- Precision@K
- Recall@K
- Mean Average Precision
- Normalized Discounted Cumulative Gain
- User relevance feedback

---

## Limitations

- The project uses a limited sample of research papers.
- Some papers may have missing abstracts, authors or topics.
- Text similarity may recommend papers containing similar words but different research objectives.
- The current deployed application does not fully use citation-network relationships.
- The model does not currently learn from individual user preferences.
- Recommendation quality may vary for papers with very short abstracts.

---

## Future Improvements

Future development could include:

- Citation-based recommendations
- A hybrid text-and-citation recommendation model
- Sentence-transformer embeddings
- A larger OpenAlex dataset
- Filters for publication year, authors, topics and citation count
- User profiles and personalized recommendations
- Recommendation evaluation metrics
- Paper bookmarking
- Exporting recommendation results
- User feedback on recommendation relevance
- Database integration for larger datasets

---

## Team Information

**Team ID:** Add the correct Team ID here

### Team Members

1. Tharun Reddy Nagula
2. Add second team member’s name
3. Add third team member’s name

---

## GitHub Repository

```text
https://github.com/tharunreddynagula-ship-it/scholarly-paper-recommender
```

---

## Live Application

Add the deployed Streamlit application link here:

```text
https://scholarly-paper-recommender.streamlit.app/
```

---

## Academic Submission

This repository contains the complete implementation and supporting files for the Machine Learning Scholarly Paper Recommendation System.

The accompanying research paper is submitted separately in **PDF format** according to the project-submission requirements.

---

## Acknowledgements

- OpenAlex for providing open scholarly metadata
- Scikit-learn for TF-IDF and cosine-similarity tools
- Streamlit for application development and deployment

---

## Disclaimer

This application was created as an academic machine learning project. Recommendations are based on textual similarity and should support, rather than replace, a complete academic literature search.
