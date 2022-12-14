FROM python:latest

WORKDIR /app

COPY /.requirements.txt /app
RUN pip install --no-cache-dir --upgrade -r requirements.txt

COPY . /app

CMD ["uvicorn", "deploying_to_cloud:app","--host", "0.0.0.0", "--port" , "8080"]