# coding: utf-8
"""Класс FastApiHandler, который обрабатывает запросы API."""
from sentence_transformers import util
import torch
import numpy as np

class FastApiHandler:
    """Класс FastApiHandler, который обрабатывает запрос и возвращает топ вопросов."""

    def __init__(self, model, corpus, corpus_embeddings):
        """Инициализация переменных класса."""
        self.model = model
        self.corpus = corpus
        self.corpus_embeddings = corpus_embeddings
        # Типы параметров запроса для проверки
        self.param_types = {
            # "question_id": str,
            "model_params": dict
        }
        # Необходимые параметры для предсказаний модели оттока
        self.required_model_params = [
            'question', 'k'
        ]

    def cos_sim_matrix(self, model_params: dict):
        """Считаем матрицу схожестей"""
        try:
            question = model_params['question']
            question_embeddings = self.model.encode([question], show_progress_bar=True)
            exclude = np.where(np.all(np.round(self.corpus_embeddings,4) == np.round(question_embeddings,4), axis=1))
            cosine_scores = util.cos_sim(question_embeddings, self.corpus_embeddings)[0]
            cosine_scores[exclude] = 0
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
                ## исправить исключение идентичного запроса (например, найти индекс такого же сообщения)
                top_values, top_indices = torch.topk(cosine_scores, k=model_params['k'])
                questions = list(self.corpus.loc[top_indices])
                top_values = top_values.tolist()
                response = {
                    # "question_id": question_id, 
                    "questions": questions,
                    "cosine_simmilarity": top_values
                }
        except Exception as e:
            print(f"Error while handling request: {e}")
            return {"Error": "Problem with request"}
        else:
            return response
