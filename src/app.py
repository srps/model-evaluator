from io import StringIO
from typing import List, Tuple
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer # TODO - Evaluate Adding TF-IDF as separate strategy
from sklearn.metrics.pairwise import cosine_similarity
from rapidfuzz import fuzz, distance
import bm25s
from bm25s.tokenization import Tokenizer
import re
import plotly.graph_objects as go
import requests
import json
import os
from sentence_transformers import SentenceTransformer

from utils import get_available_st_models, download_model, timing_decorator

st.set_page_config(layout="wide")

tf_serving_url = st.sidebar.text_input("TensorFlow Serving URL", "http://tf-serving:8501")
tf_model_name = st.sidebar.text_input("TensorFlow Model Name", "muse")


with st.sidebar:
    # Option to download new models
    st.subheader("Download New Models")
    available_models = get_available_st_models()
    if available_models:
        st.info("You can download additional Sentence Transformer models.")
    else:
        st.warning("No Sentence Transformer models found. You can download models to the 'models' directory.")
    model_name = st.text_input("Enter the name of the Sentence Transformer model to download:")
    if st.button("Download Model"):
        download_model(model_name)
        st.success(f"Model '{model_name}' has been downloaded. Please refresh the page to use it.")
        st.rerun()
        

# Session state variables
def initialize_session_state():
    if 'text_embeddings' not in st.session_state:
        st.session_state.text_embeddings = {}
    if 'current_model' not in st.session_state:
        st.session_state.current_model = None
    if 'current_texts' not in st.session_state:
        st.session_state.current_texts = None
    if 'preprocessed_texts' not in st.session_state:
        st.session_state.preprocessed_texts = None
    if 'bm25_retriever' not in st.session_state:
        st.session_state.bm25_retriever = None
    if 'tokenizer' not in st.session_state:
        st.session_state.tokenizer = None
    if 'query' not in st.session_state:
        st.session_state.query = None
        

# Preprocessing function
# TODO: Add customization options for preprocessing
def preprocess(text: str) -> str:
    return re.sub(r'[^\w\s]', '', text.lower())

# Exact Match
@timing_decorator
def exact_match(query: str, texts: list) -> list:
    return [1.0 if query == text else 0.0 for text in texts]

# Prefix Match
@timing_decorator
def prefix_match(query: str, texts: list) -> list:
    return [1.0 if distance.Prefix.similarity(query, text) == len(query) else 0.0 for text in texts]

# Fuzzy Match
@timing_decorator
def fuzzy_match(query: str, texts: list) -> list:
    threshold = 80
    return [ratio / 100 if (ratio := fuzz.ratio(query, text)) >= threshold else 0. for text in texts]

# BM25 Retrieval
@st.cache_resource
def create_bm25_index(texts: list, _tokenizer: Tokenizer) -> Tuple[bm25s.BM25, List[List[str]]]:
    tokenized_texts = _tokenizer.tokenize(texts, update_vocab=True)
    corpus = { i: text for i, text in enumerate(texts) }
    retriever = bm25s.BM25(corpus=corpus)
    retriever.index(tokenized_texts)
    return retriever, tokenized_texts

# BM25 Retrieval
# TODO: Add options for fuzzy / prefix matching to BM25
@timing_decorator
def bm25_retrieval(query: str, retriever: bm25s.BM25, tokenizer: Tokenizer) -> np.ndarray:
    query_tokens = tokenizer.tokenize([query], update_vocab=False)
    scores_array = np.zeros(len(retriever.corpus))
    scores_array += retriever.get_scores_from_ids(query_tokens[0])
    min_score = np.min(scores_array)
    max_score = np.max(scores_array)
    normalized_scores = (scores_array - min_score) / (max_score - min_score if max_score != min_score else 1.0)
    return normalized_scores

# Universal Sentence Encoder (TF Serving)
def tf_encode(texts: list, model_name: str, base_url: str = "http://tf-serving:8501") -> np.ndarray:
    url = f"{base_url}/v1/models/{model_name}:predict"
    data = json.dumps({"instances": texts})
    response = requests.post(url, data=data)
    response.raise_for_status()
    predictions = response.json()['predictions']
    return np.array(predictions)

# Sentence Transformer
# TODO: Add option to run inference via API or locally
@st.cache_resource
def load_sentence_transformer(model_name):
    return SentenceTransformer(model_name)

def calculate_embeddings(texts: list, model: SentenceTransformer, model_name: str):
    total_texts = len(texts)
    embeddings = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    status_text.text(f"Calculating embeddings for {total_texts} texts using {model_name}")
    
    for i, text in enumerate(texts):
        if model_name == tf_model_name:
            embeddings.append(tf_encode(texts=[text], model_name=model_name, base_url=tf_serving_url)[0])
        else:
            embeddings.append(model.encode([text])[0])
        
        progress = (i + 1) / total_texts
        progress_bar.progress(progress, f"{progress:.1%}")
    
    embeddings = np.array(embeddings)
    st.session_state.text_embeddings[model_name] = embeddings
    st.session_state.current_model = model_name
    st.session_state.current_texts = texts
    
    status_text.text(f"Embedding calculation complete for {total_texts} texts using {model_name}")

# Semantic Search
@timing_decorator
def semantic_search(query, texts, model: SentenceTransformer | None, model_name: str)  -> list:
    if st.session_state.text_embeddings[model_name] is None:
        calculate_embeddings(texts, model, model_name)
    if model_name == tf_model_name:
        query_embedding = tf_encode([query], model_name=model_name, base_url=tf_serving_url)[0]
    else:
        query_embedding = model.encode([query])[0]
    
    similarities: np.ndarray = cosine_similarity([query_embedding], st.session_state.text_embeddings[model_name])[0]
    return similarities.tolist()

@timing_decorator
def semantic_search_multi(query: str, texts: list, models: dict[str, SentenceTransformer | None]) -> dict[str, list]:
    results = {}
    for model_name, model in models.items():
        if model_name not in st.session_state.text_embeddings:
            calculate_embeddings(texts, {model_name: model})
        if model_name == tf_model_name:
            query_embedding = tf_encode([query])[0]
        else:
            query_embedding = model.encode([query])[0]
        
        similarities: np.ndarray = cosine_similarity([query_embedding], st.session_state.text_embeddings[model_name])[0]
        results[model_name] = similarities.tolist()
    return results

# Visualization function
def create_timing_chart(timing_data: dict) -> go.Figure:
    fig = go.Figure(data=[
        go.Bar(
            x=list(timing_data.keys()),
            y=list(timing_data.values()),
            text=[f"{value:.4f}s" for value in timing_data.values()],
            textposition='auto',
        )
    ])
    fig.update_layout(
        title="Execution Time for Each Search Strategy",
        xaxis_title="Search Strategy",
        yaxis_title="Time (seconds)",
        height=500
    )
    return fig

def upload_callback():
    # Clear existing embeddings when new file is uploaded
    st.session_state.text_embeddings = {}
    st.session_state.current_model = None
    st.session_state.current_texts = None
    st.session_state.bm25_retriever = None
    st.session_state.query = None

# Main Streamlit app
def main():
    st.title("Embedding Model and Search Strategy Evaluator")

    # Initialize session state
    initialize_session_state()
    
    # Create two columns
    left_column, right_column = st.columns([1, 3])

    with left_column:
        st.subheader("Search Strategies")
        # File uploader for corpus
        uploaded_file = st.file_uploader("Upload Corpus (TXT)", type="txt", on_change=upload_callback, key="file_uploader")

        if uploaded_file is not None:
            stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
            string_data = stringio.read()
            texts = string_data.splitlines()
            preprocessed_texts = [preprocess(text) for text in texts]
            
            # Create Tokenizer
            if st.session_state.tokenizer is None:
                st.session_state.tokenizer = Tokenizer()

            # Create BM25 index
            if st.session_state.bm25_retriever is None:
                bm25_retriever, tokenized_texts = create_bm25_index(preprocessed_texts, st.session_state.tokenizer)
                st.session_state.bm25_retriever = bm25_retriever

            # Model selection
            model_options = [tf_model_name] + get_available_st_models()
            selected_models = st.multiselect("Select models to evaluate", model_options, default=[tf_model_name])
            
            models = {}
            for model_name in selected_models:
                if model_name == tf_model_name:
                    models[model_name] = None
                else:
                    models[model_name] = load_sentence_transformer(os.path.join("models", model_name))

            if not models:
                st.warning("No models selected. Please select at least one model to evaluate.")
            else:
                for model_name, model in models.items():
                    if model_name == tf_model_name:
                        st.info(f"Using {model_name} via TensorFlow Serving")
                        if model_name not in st.session_state.text_embeddings:
                            calculate_embeddings(texts, None, model_name)
                    else:
                        st.info(f"Using {model_name}")
                        if model_name not in st.session_state.text_embeddings:
                            calculate_embeddings(texts, model, model_name)

            # Search input
            st.session_state.query = st.text_input("Enter your search query:")
            preprocessed_query = preprocess(st.session_state.query)

            if st.session_state.query:
                # Select search strategies and set weights
                search_strategies = {
                    "Exact Match": st.checkbox("Exact Match", value=True),
                    "Prefix Match": st.checkbox("Prefix Match", value=True),
                    "Fuzzy Match": st.checkbox("Fuzzy Match", value=True),
                    "BM25 Retrieval": st.checkbox("BM25 Retrieval", value=True),
                    "Semantic Search": st.checkbox("Semantic Search", value=True)
                }

                weights = {}
                for strategy, selected in search_strategies.items():
                    if selected:
                        weights[strategy] = st.slider(f"{strategy} Weight", 0.0, 1.0, 1.0, 0.1)
                        
        with right_column:
            st.subheader("Search Results")
            
            if st.session_state.query:
                st.write(f"Query: {st.session_state.query}")
            
                results = []
                timing_data = {}
                semantic_scores = {}

                # TODO - Add Rank Fusion Strategies and cleanup this section
                if search_strategies["Exact Match"]:
                    exact_scores, exact_time = exact_match(preprocessed_query, preprocessed_texts)
                    timing_data["Exact Match"] = exact_time
                if search_strategies["Prefix Match"]:
                    prefix_scores, prefix_time = prefix_match(preprocessed_query, preprocessed_texts)
                    timing_data["Prefix Match"] = prefix_time
                if search_strategies["Fuzzy Match"]:
                    fuzzy_scores, fuzzy_time = fuzzy_match(preprocessed_query, preprocessed_texts)
                    timing_data["Fuzzy Match"] = fuzzy_time
                if search_strategies["BM25 Retrieval"]:
                    bm25_scores, bm25_time = bm25_retrieval(preprocessed_query, st.session_state.bm25_retriever, st.session_state.tokenizer)
                    timing_data["BM25 Retrieval"] = bm25_time
                if search_strategies["Semantic Search"]:
                    for model_name, model in models.items():
                        st.write(f"Using {model_name}")
                        semantic_scores[model_name], semantic_time = semantic_search(st.session_state.query, texts, model, model_name)
                        timing_data[f"STS ({model_name})"] = semantic_time

                for i, text in enumerate(texts):
                    result = {"Text": text}
                    total_weight = 0
                    weighted_score = 0

                    if search_strategies["Exact Match"]:
                        result["Exact Match"] = exact_scores[i]
                        weighted_score += exact_scores[i] * weights["Exact Match"]
                        total_weight += weights["Exact Match"]
                    if search_strategies["Prefix Match"]:
                        result["Prefix Match"] = prefix_scores[i]
                        weighted_score += prefix_scores[i] * weights["Prefix Match"]
                        total_weight += weights["Prefix Match"]
                    if search_strategies["Fuzzy Match"]:
                        result["Fuzzy Match"] = fuzzy_scores[i]
                        weighted_score += fuzzy_scores[i] * weights["Fuzzy Match"]
                        total_weight += weights["Fuzzy Match"]
                    if search_strategies["BM25 Retrieval"]:
                        result["BM25 Retrieval"] = bm25_scores[i]
                        weighted_score += bm25_scores[i] * weights["BM25 Retrieval"]
                        total_weight += weights["BM25 Retrieval"]
                    if search_strategies["Semantic Search"]:
                        semantic_weight = weights["Semantic Search"] / len(models)  # Distribute weight among models
                        for model_name in models:
                            result[f"STS ({model_name})"] = semantic_scores[model_name][i]
                            weighted_score += semantic_scores[model_name][i] * semantic_weight
                            total_weight += semantic_weight

                    result["Overall Score"] = weighted_score / total_weight if total_weight > 0 else 0
                    results.append(result)

                results_df = pd.DataFrame(results).sort_values("Overall Score", ascending=False).head(10)
                # Reorder columns to group semantic search results together
                columns = ["Text"] + [col for col in results_df.columns if col.startswith("STS")] + [col for col in results_df.columns if not col.startswith("STS") and col != "Text"]
                results_df = results_df[columns]

                st.write(results_df)

                # Display timing information
                st.subheader("Execution Time")
                st.write(f"Total execution time: {sum(timing_data.values()):.4f} seconds")

                # Create and display timing chart
                timing_chart = create_timing_chart(timing_data)
                st.plotly_chart(timing_chart)

if __name__ == "__main__":
    main()