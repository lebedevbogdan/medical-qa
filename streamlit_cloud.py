import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sentence_transformers import util, SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"
DATA_PATH = "embeddings.pkl"
MAX_VALUE = 10

@st.cache_data
def make_embeddings(path):
    model = SentenceTransformer(MODEL_NAME)
    # Load sentences & embeddings from disc
    with open(path, "rb") as fIn:
        stored_data = pickle.load(fIn)
    corpus = stored_data["sentences"]
    corpus_embeddings = stored_data["embeddings"]
    return model, corpus, corpus_embeddings

model, corpus, corpus_embeddings = make_embeddings(DATA_PATH)

@st.cache_data
def return_df(_model, question):
    question_embeddings = model.encode([question], show_progress_bar=True)
    exclude = np.where(np.all(np.round(corpus_embeddings,4) == np.round(question_embeddings,4), axis=1))
    cosine_scores = util.cos_sim(question_embeddings, corpus_embeddings)[0]
    cosine_scores[exclude] = 0
    sorted_indices = np.argsort(-cosine_scores)
    top_indices = sorted_indices[:MAX_VALUE]
    top_values = cosine_scores[top_indices]
    questions = list(corpus.loc[top_indices])
    df = pd.DataFrame({'questions': questions, 'cosine_simmilarity': top_values})
    return df

if 'stage' not in st.session_state:
    st.session_state.stage = 0
def set_state(i):
    st.session_state.stage = i

st.title('Similar question search')
text_input = st.text_input('Ask a question:', 'After how many hour from drinking an antibiotic can I drink alcohol?', on_change=return_df.clear())
k = st.slider("Select a number of options", value=5, min_value=1, max_value=MAX_VALUE)


if st.session_state.stage == 0:
    st.button('Find', on_click=set_state, args=[1])

if st.session_state.stage >= 1:       
    df = return_df(model, text_input)
    set_state(2)
    
if st.session_state.stage >= 2:       
    if text_input != '':
        st.dataframe(df.head(k))
    else:
        st.write('You should write a question.')
