# medical-qa

Цель проекта - найти по введенному вопросу наиболее близкие по смыслу вопросы из подготовленного списка.

Для данной задачи произведено сравнение классически nlp методов с методами, использующие трансформеры. 

Метрикой сравнения является ACCURACY@N. Для вопросов из списка есть размеченные пары, которые указывают на схожесть вопросов (label = 1).
Если среди подобранных топ-N вопросов будет тот, который в исходных данных помечен как label=1, то считаем, что предсказание - верное.

Исследование в джупитер-ноутбуке: [encoders.ipynb](https://github.com/lebedevbogdan/medical-qa/blob/main/encoders.ipynb). По результатам исследования для создания эмбедингов был выбран чекпоинт [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)

На основании выбранной модели были собраны два варианта приложения:
1. Приложение для загрузки в облако Streamlit ([ссылка на приложение](https://medical-app-q.streamlit.app)). Код приложения в файле [streamlit_cloud.py](https://github.com/lebedevbogdan/medical-qa/blob/main/streamlit_cloud.py)
2. Компоненты приложения, собранные в докер-образы. Отдельно [frontend](https://github.com/lebedevbogdan/medical-qa/tree/main/frontend) и [backend](https://github.com/lebedevbogdan/medical-qa/tree/main/backend)

Для backend микросервиса использовались **uvicorn** и **FastAPI**, для frontend - **Streamlit**

Для запуска приложения необходимо из корневой папки проека выполнить команду:
```
docker compose up --build
```
Файл [Docker Compose](https://github.com/lebedevbogdan/medical-qa/blob/main/docker-compose.yaml)

