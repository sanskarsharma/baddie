FROM python:3.9-slim

RUN apt-get update \
    && apt-get install -y build-essential curl software-properties-common \
    && rm -rf /var/lib/apt/lists/*

WORKDIR .
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY streamlit_app.py streamlit_app.py
COPY ./libs libs
COPY ./llm llm

EXPOSE 6969
ENTRYPOINT ["streamlit", "run", "streamlit_app.py", "--server.port=6969", "--server.fileWatcherType=none"]
