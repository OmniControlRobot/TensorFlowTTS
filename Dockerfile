FROM python:3.7

RUN apt-get update
RUN pip install -U pip setuptools

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 80

CMD python app.py