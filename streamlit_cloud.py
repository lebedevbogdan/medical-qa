import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util

MODEL_NAME = "all-MiniLM-L6-v2"
DATA_PATH = "medical_questions_pairs.csv"

@st.cache_data
def make_embeddings(model_name, path):
    model = SentenceTransformer(model_name)
    data = pd.read_csv(path)
    corpus = pd.concat([data['question_1'], data['question_2']]).drop_duplicates().reset_index(drop=True)
    corpus_embeddings = model.encode(corpus, show_progress_bar=True)
    return model, corpus, corpus_embeddings

model, corpus, corpus_embeddings = make_embeddings(MODEL_NAME, DATA_PATH)

if 'stage' not in st.session_state:
    st.session_state.stage = 0
def set_state(i):
    st.session_state.stage = i

st.title('Similar question search')
text_input = st.text_input('Ask a question:', 'After how many hour from drinking an antibiotic can I drink alcohol?')
k = st.slider("Select a number of options", value=5, min_value=1, max_value=10)


if st.session_state.stage == 0:
    st.button('Find', on_click=set_state, args=[1])

def return_df(model, question):
    question_embeddings = model.encode([question], show_progress_bar=True)
    exclude = np.where(np.all(np.round(corpus_embeddings,4) == np.round(question_embeddings,4), axis=1))
    cosine_scores = util.cos_sim(question_embeddings, corpus_embeddings)[0]
    cosine_scores[exclude] = 0
    sorted_indices = np.argsort(-cosine_scores)
    top_indices = sorted_indices[:k]
    top_values = cosine_scores[top_indices]
    questions = list(corpus.loc[top_indices])
    df = pd.DataFrame({'questions': questions, 'cosine_simmilarity': top_values})
    return df

if st.session_state.stage >= 1:       
    df = return_df(model, text_input)
    st.dataframe(df)
