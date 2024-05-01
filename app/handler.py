# coding: utf-8
"""Класс FastApiHandler, который обрабатывает запросы API."""
from sentence_transformers import SentenceTransformer, util
import torch
import pandas as pd

class FastApiHandler:
    """Класс FastApiHandler, который обрабатывает запрос и возвращает топ вопросов."""

    def __init__(self):
        """Инициализация переменных класса."""

        # Типы параметров запроса для проверки
        self.param_types = {
            # "question_id": str,
            "model_params": dict
        }

        self.model_name = "all-MiniLM-L6-v2"
        self.data_path = "../medical_questions_pairs.csv"
        self.load_embedding_model(model_path=self.model_name)
        self.data = pd.read_csv(self.data_path)
        self.corpus = pd.concat([self.data['question_1'], self.data['question_2']]).drop_duplicates().reset_index(drop=True)
        # Необходимые параметры для предсказаний модели оттока
        self.required_model_params = [
            'sentence', 'k'
        ]

    def load_embedding_model(self, model_path: str):
        """Загружаем обученную модель SentenceTransformer.
        Args:
            model_path (str): Путь до модели.
        """
        try:
            self.model = SentenceTransformer(model_path)

        except Exception as e:
            print(f"Failed to load model: {e}")

    def cos_sim_matrix(self, model_params: dict):
        """Считаем матрицу схожестей"""
        try:
            corpus_embeddings = self.model.encode(self.corpus, show_progress_bar=True)
            sentence = model_params['sentence']
            sentence_embeddings = self.model.encode(sentence, show_progress_bar=True)
            cosine_scores = util.cos_sim(sentence_embeddings, corpus_embeddings)
            return cosine_scores
        except Exception as e:
            print(f"Failed to count matrix: {e}")
        
    def check_required_query_params(self, query_params: dict) -> bool:
        """Проверяем параметры запроса на наличие обязательного набора параметров.
        
        Args:
            query_params (dict): Параметры запроса.
        
        Returns:
                bool: True - если есть нужные параметры, False - иначе
        """
        # if "question_id" not in query_params or "model_params" not in query_params:
        #     return False
        
        # if not isinstance(query_params["question_id"], self.param_types["question_id"]):
        #     return False
                
        if not isinstance(query_params["model_params"], self.param_types["model_params"]):
            return False
        return True
    
    def check_required_model_params(self, model_params: dict) -> bool:
        """Проверяем параметры пользователя на наличие обязательного набора.
    
        Args:
            model_params (dict): Параметры пользователя для предсказания.
    
        Returns:
            bool: True - если есть нужные параметры, False - иначе
        """
        if set(model_params.keys()) == set(self.required_model_params):
            return True
        return False
    
    def validate_params(self, params: dict) -> bool:
        """Разбираем запрос и проверяем его корректность.
    
        Args:
            params (dict): Словарь параметров запроса.
    
        Returns:
            - **dict**: Cловарь со всеми параметрами запроса.
        """
        if self.check_required_query_params(params):
            print("All query params exist")
        else:
            print("Not all query params exist")
            return False
        
        if self.check_required_model_params(params["model_params"]):
            print("All model params exist")
        else:
            print("Not all model params exist")
            return False
        return True
		
    def handle(self, params):
        """Функция для обработки запросов API параметров входящего запроса.
    
        Args:
            params (dict): Словарь параметров запроса.
    
        Returns:
            - **dict**: Словарь, содержащий результат выполнения запроса.
        """
        try:
            # Валидируем запрос к API
            if not self.validate_params(params):
                print("Error while handling request")
                response = {"Error": "Problem with parameters"}
            else:
                model_params = params["model_params"]
                # question_id = params["question_id"]
                # print(f"Predicting for question_id: {question_id} and model_params:\n{model_params}")
                # Получаем предсказания модели
                cosine_scores = self.cos_sim_matrix(model_params)
                top_values, top_indices = torch.topk(cosine_scores[cosine_scores<2], k=model_params['k']+1)
                sentences = list(self.corpus.loc[top_indices])[1:]
                top_values = top_values.tolist()[1:]
                response = {
                    # "question_id": question_id, 
                    "cosine_simmilarity": top_values, 
                    "sentences": sentences
                }
        except Exception as e:
            print(f"Error while handling request: {e}")
            return {"Error": "Problem with request"}
        else:
            return response

if __name__ == "__main__":

    # Создаем тестовый запрос
    test_params = {
	    # "question_id": "101",
        "model_params": {
            'k':3,
            'sentence': "After how many hour from drinking an antibiotic can I drink alcohol?"
        }
    }

    # Создаем обработчик запросов для API
    handler = FastApiHandler()

    # Делаем тестовый запрос
    response = handler.handle(test_params)
    print(f"Response: {response}")