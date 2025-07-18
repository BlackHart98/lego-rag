FROM python:3.11-slim

WORKDIR /chroma_database

COPY requirements.txt .

EXPOSE 8000

CMD ["chroma", "run", "--path", "/chroma_database"]