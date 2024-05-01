"""Приложение Fast API для поиска ближайших вопросов."""


from fastapi import FastAPI, Body
from handler import FastApiHandler


"""
Пример запуска из директории mle-sprint3/app:
uvicorn qa_app:app --reload --port 8081 --host 0.0.0.0

Для просмотра документации API и совершения тестовых запросов зайти на  http://127.0.0.1:8081/docs

Если используется другой порт, то заменить 8081 на этот порт
"""


# Создаем приложение Fast API
app = FastAPI()

# Создаем обработчик запросов для API
app.handler = FastApiHandler()


@app.post("/api/qa/") 
def get_simmilar_docs(
    # question_id: str,
    model_params: dict = Body(
        example={
            'k':3,
            'sentence': "After how many hour from drinking an antibiotic can I drink alcohol?"
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