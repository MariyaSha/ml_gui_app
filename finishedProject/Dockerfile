FROM python:3.9-slim
MAINTAINER Label="Rohan Rustagi"
WORKDIR /app
COPY . /app
RUN pip install --no-cache-dir -r requirements.txt
EXPOSE 5000
CMD ["python3", "classifier.py"]
