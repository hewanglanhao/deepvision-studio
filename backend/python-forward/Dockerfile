FROM python:3.12-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .

ENV FORWARD_HOST=0.0.0.0 \
    FORWARD_PORT=5000 \
    FORWARD_DEBUG=false

EXPOSE 5000
CMD ["python", "app.py"]
