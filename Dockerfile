FROM python:3.10-slim

WORKDIR /app


RUN apt-get update && apt-get install ffmpeg libsm6 libxext6 git  -y

COPY requirements.txt .

RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

COPY . .

ENTRYPOINT [ "bash" ]