FROM gcr.io/tensorflow/tensorflow:1.5.0-gpu-py3
RUN pip install joblib==0.11 gensim==3.2.0
RUN mkdir /app

WORKDIR /app/
COPY . /app/
VOLUME ["/app/input", "/app/models/"]
CMD ["python", "train_deep.py", "/app/input/questions_1000.txt"]
