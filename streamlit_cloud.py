import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sentence_transformers import util, SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"
DATA_PATH = "embeddings.pkl"
MAX_VALUE = 10

@st.cache_resource(show_spinner="Загружаю модель")
def make_embeddings(path):
    model = SentenceTransformer(MODEL_NAME)
    with open(path, "rb") as fIn:
        stored_data = pickle.load(fIn)
    corpus = stored_data["sentences"]
    corpus_embeddings = stored_data["embeddings"]
    return model, corpus, corpus_embeddings

model, corpus, corpus_embeddings = make_embeddings(DATA_PATH)

@st.cache_data
def return_df(_model, question):
    question_embeddings = model.encode([question], show_progress_bar=True)
    # Находим индекс в тензоре схожестей, который соответствует тому же предложению (то есть схожесть равна 1)
    exclude = np.where(np.all(np.round(corpus_embeddings,4) == np.round(question_embeddings,4), axis=1))
    cosine_scores = util.cos_sim(question_embeddings, corpus_embeddings)[0]
    # Чтобы не учитывать то же самое предложение, назначаем ему схожесть 0
    cosine_scores[exclude] = 0
    # Сортируем схожести по убыванию и осталвяем первые MAX_VALUE значений
    sorted_indices = np.argsort(-cosine_scores)
    top_indices = sorted_indices[:MAX_VALUE]
    top_values = cosine_scores[top_indices]
    # Собираем предложения в датафрейм
    questions = list(corpus.loc[top_indices])
    df = pd.DataFrame({'questions': questions, 'cosine_similarity': top_values})
    return df

if 'stage' not in st.session_state:
    st.session_state.stage = 0
def set_state(i):
    st.session_state.stage = i

st.title('Similar question search')
text_input = st.text_input('Ask a question:', 'After how many hour from drinking an antibiotic can I drink alcohol?')
k = st.slider("Select a number of options", value=5, min_value=1, max_value=MAX_VALUE)

# При запуске отобразится кнопка Find, которая при нажатии переведен в состояние 1
if st.session_state.stage == 0:
    st.button('Find', on_click=set_state, args=[1])

if st.session_state.stage >= 1:       
    df = return_df(model, text_input)
    set_state(2)
    
# Если в строке ввода пусто и запущен поиск, тогда вывести предупреждение
if st.session_state.stage >= 2:       
    if text_input != '':
        st.dataframe(df.head(k))
    else:
        st.markdown('You should write a question')
