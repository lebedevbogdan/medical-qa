"""Приложение Fast API для поиска ближайших вопросов."""
from sentence_transformers import SentenceTransformer
import pandas as pd

from fastapi import FastAPI, Body
from handler import FastApiHandler

MODEL_NAME = "all-MiniLM-L6-v2"
DATA_PATH = "../medical_questions_pairs.csv"

model = SentenceTransformer(MODEL_NAME)
data = pd.read_csv(DATA_PATH)

corpus = pd.concat([data['question_1'], data['question_2']]).drop_duplicates().reset_index(drop=True)
corpus_embeddings = model.encode(corpus, show_progress_bar=True)

# Создаем приложение Fast API
app = FastAPI()

# Создаем обработчик запросов для API
app.handler = FastApiHandler(model=model, corpus=corpus, corpus_embeddings=corpus_embeddings)

@app.post("/api/qa/") 
def get_simmilar_docs(
    # question_id: str,
    model_params: dict = Body(
        example={
            'k':3,
            'question': "After how many hour from drinking an antibiotic can I drink alcohol?"
        }
    )
):
    """Функция для получения косинусных схожестей вопросов.

    Args:
        question_id (str): Идентификатор вопроса.
        model_params (dict): Параметры вопроса, которые мы должны подать в модель.

    Returns:
        dict: Топ-N схожих вопросов.
    """
    all_params = {
        # "question_id": question_id,
        "model_params": model_params
    }
    return app.handler.handle(all_params)
