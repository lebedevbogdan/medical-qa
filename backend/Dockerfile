FROM python:3.10-slim

COPY requirements.txt app/requirements.txt
WORKDIR app

RUN apt-get update && apt-get install -y gcc python3-dev && apt-get clean && rm -rf /var/lib/apt/lists/
RUN pip3 install -r requirements.txt
COPY . /app
EXPOSE 8000

CMD uvicorn main:app --host 0.0.0.0 --port 8000 --reload