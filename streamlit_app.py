import streamlit as st
import requests
import pandas as pd

if 'stage' not in st.session_state:
    st.session_state.stage = 0
def set_state(i):
    st.session_state.stage = i

# Заголовок приложения
st.title('Similar question search')

# Поле для ввода текста
text_input = st.text_input('Ask a question:', 'After how many hour from drinking an antibiotic can I drink alcohol?')

# Поле для ввода кол-ва
number = st.slider("Select a number of options", value=5, min_value=1, max_value=10)

# Кнопка для отправки POST-запроса
if st.session_state.stage == 0:
    st.button('Find', on_click=set_state, args=[1])
# URL вашего микросервиса
url = 'http://127.0.0.1:8081/api/qa/'
# Тело POST-запроса
data = {
# "question_id": "101",
# "model_params": {
    'k':number,
    'question': text_input
    }
# }
if st.session_state.stage >= 1:
    # Отправка POST-запроса
    response = requests.post(url, json=data)
    print(response)
    # Вывод ответа микросервиса
    st.text('Response:')
    # st.json(response.json())
    st.dataframe(pd.DataFrame(response.json()))