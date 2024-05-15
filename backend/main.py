"""Приложение Fast API для поиска ближайших вопросов."""
from sentence_transformers import SentenceTransformer
import pickle

from fastapi import FastAPI, Body
from handler import FastApiHandler

MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDINGS_PATH = "embeddings.pkl"
model = SentenceTransformer(MODEL_NAME)

model = SentenceTransformer(MODEL_NAME)
# Открываем подготовленные эмбеддинги с предложениями для более быстрого запуска приложения
with open(EMBEDDINGS_PATH, "rb") as fIn:
    stored_data = pickle.load(fIn)
    corpus = stored_data["sentences"]
    corpus_embeddings = stored_data["embeddings"]

# Создаем приложение Fast API
app = FastAPI()

# Создаем обработчик запросов для API
app.handler = FastApiHandler(model=model, corpus=corpus, corpus_embeddings=corpus_embeddings)

@app.get("/")
async def read_root():
    return {"message": "Welcome to the model API!"}

@app.post("/qa/") 
async def get_simmilar_docs(
    # question_id: str,
    model_params: dict = Body(
        example={
            'k':3,
            'question': "I have a toothache, what should I do?"
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
