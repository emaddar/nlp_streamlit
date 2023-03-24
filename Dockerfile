# app/Dockerfile

FROM python:3.9-slim

WORKDIR /app

ENV PYTHONUNBUFFERED 1

COPY . /app

RUN pip install --upgrade pip && \
    pip install -r requirements.txt

EXPOSE 8000

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["streamlit", "run", "nlp_sreamlit.py", "--server.port=8501", "--server.address=0.0.0.0"]

# docker build -t nlp .
# docker run -p 8501:8501 nlp