# Руководство по развертыванию RAG системы

## Локальное развертывание

### Вариант 1: Автоматический (рекомендуется)

#### Windows
```bash
# Двойной клик на run.bat или:
run.bat
```

#### Linux/Mac
```bash
chmod +x run.sh
./run.sh
```

Скрипт автоматически:
1. Создаст виртуальное окружение
2. Установит зависимости
3. Построит индекс (если нужно)
4. Запустит API сервер

### Вариант 2: Ручной

```bash
# 1. Создание виртуального окружения
python -m venv venv

# 2. Активация
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# 3. Установка зависимостей
pip install -r requirements.txt

# 4. Построение индекса
python build_index.py

# 5. Запуск API
python api.py
```

## Docker развертывание

### Создание Dockerfile

```dockerfile
# Dockerfile
FROM python:3.9-slim

# Установка системных зависимостей
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# Рабочая директория
WORKDIR /app

# Копирование зависимостей
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копирование кода
COPY . .

# Порт
EXPOSE 8000

# Команда запуска
CMD ["python", "api.py"]
```

### Создание docker-compose.yml

```yaml
version: '3.8'

services:
  rag-api:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./index:/app/index
      - ../dataset_documents:/app/dataset_documents:ro
    environment:
      - DOCUMENTS_PATH=/app/dataset_documents
      - INDEX_PATH=/app/index
      - API_HOST=0.0.0.0
      - API_PORT=8000
    restart: unless-stopped
```

### Запуск с Docker

```bash
# Построение образа
docker-compose build

# Построение индекса (один раз)
docker-compose run --rm rag-api python build_index.py

# Запуск сервиса
docker-compose up -d

# Проверка логов
docker-compose logs -f

# Остановка
docker-compose down
```

## Облачное развертывание

### AWS EC2

#### 1. Создание инстанса

```bash
# Выберите:
# - Instance type: t3.medium или больше (4GB+ RAM)
# - AMI: Ubuntu 22.04 LTS
# - Storage: 20GB+
# - Security Group: открыть порт 8000
```

#### 2. Подключение и настройка

```bash
# SSH подключение
ssh -i your-key.pem ubuntu@your-instance-ip

# Обновление системы
sudo apt update && sudo apt upgrade -y

# Установка зависимостей
sudo apt install -y python3-pip python3-venv tesseract-ocr poppler-utils git

# Клонирование проекта
git clone your-repo-url
cd rag_ml

# Установка Python зависимостей
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Копирование документов
scp -i your-key.pem -r dataset_documents ubuntu@your-instance-ip:~/

# Построение индекса
python build_index.py

# Запуск с nohup
nohup python api.py > api.log 2>&1 &
```

#### 3. Настройка systemd сервиса

```bash
# Создание сервиса
sudo nano /etc/systemd/system/rag-api.service
```

```ini
[Unit]
Description=RAG API Service
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/rag_ml
Environment="PATH=/home/ubuntu/rag_ml/venv/bin"
ExecStart=/home/ubuntu/rag_ml/venv/bin/python api.py
Restart=always

[Install]
WantedBy=multi-user.target
```

```bash
# Запуск сервиса
sudo systemctl daemon-reload
sudo systemctl enable rag-api
sudo systemctl start rag-api

# Проверка статуса
sudo systemctl status rag-api

# Просмотр логов
sudo journalctl -u rag-api -f
```

### Google Cloud Platform (GCP)

#### 1. Создание VM

```bash
gcloud compute instances create rag-api-vm \
  --machine-type=e2-medium \
  --image-family=ubuntu-2204-lts \
  --image-project=ubuntu-os-cloud \
  --boot-disk-size=20GB \
  --tags=http-server
```

#### 2. Настройка firewall

```bash
gcloud compute firewall-rules create allow-rag-api \
  --allow=tcp:8000 \
  --target-tags=http-server
```

#### 3. Подключение и настройка

```bash
gcloud compute ssh rag-api-vm

# Далее аналогично AWS EC2
```

### Azure

#### 1. Создание VM

```bash
az vm create \
  --resource-group myResourceGroup \
  --name rag-api-vm \
  --image UbuntuLTS \
  --size Standard_B2s \
  --admin-username azureuser \
  --generate-ssh-keys
```

#### 2. Открытие порта

```bash
az vm open-port --port 8000 --resource-group myResourceGroup --name rag-api-vm
```

#### 3. Подключение и настройка

```bash
ssh azureuser@your-vm-ip

# Далее аналогично AWS EC2
```

## Nginx reverse proxy

### Установка Nginx

```bash
sudo apt install nginx
```

### Конфигурация

```bash
sudo nano /etc/nginx/sites-available/rag-api
```

```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

```bash
# Активация конфигурации
sudo ln -s /etc/nginx/sites-available/rag-api /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx
```

### SSL с Let's Encrypt

```bash
# Установка certbot
sudo apt install certbot python3-certbot-nginx

# Получение сертификата
sudo certbot --nginx -d your-domain.com

# Автоматическое обновление
sudo certbot renew --dry-run
```

## Мониторинг

### Prometheus + Grafana

#### 1. Добавление метрик в API

```python
# api.py
from prometheus_client import Counter, Histogram, generate_latest
from fastapi import Response

# Метрики
request_count = Counter('rag_requests_total', 'Total requests')
request_duration = Histogram('rag_request_duration_seconds', 'Request duration')

@app.get("/metrics")
async def metrics():
    return Response(generate_latest(), media_type="text/plain")

@app.middleware("http")
async def add_metrics(request, call_next):
    request_count.inc()
    with request_duration.time():
        response = await call_next(request)
    return response
```

#### 2. Конфигурация Prometheus

```yaml
# prometheus.yml
scrape_configs:
  - job_name: 'rag-api'
    static_configs:
      - targets: ['localhost:8000']
```

#### 3. Запуск с Docker Compose

```yaml
# docker-compose.monitoring.yml
version: '3.8'

services:
  prometheus:
    image: prom/prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml

  grafana:
    image: grafana/grafana
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
```

## Масштабирование

### Горизонтальное масштабирование

#### 1. Load Balancer (Nginx)

```nginx
upstream rag_backend {
    server 127.0.0.1:8000;
    server 127.0.0.1:8001;
    server 127.0.0.1:8002;
}

server {
    listen 80;
    
    location / {
        proxy_pass http://rag_backend;
    }
}
```

#### 2. Запуск нескольких инстансов

```bash
# Инстанс 1
API_PORT=8000 python api.py &

# Инстанс 2
API_PORT=8001 python api.py &

# Инстанс 3
API_PORT=8002 python api.py &
```

### Kubernetes

#### 1. Deployment

```yaml
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: rag-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: rag-api
  template:
    metadata:
      labels:
        app: rag-api
    spec:
      containers:
      - name: rag-api
        image: your-registry/rag-api:latest
        ports:
        - containerPort: 8000
        resources:
          requests:
            memory: "2Gi"
            cpu: "1"
          limits:
            memory: "4Gi"
            cpu: "2"
```

#### 2. Service

```yaml
# service.yaml
apiVersion: v1
kind: Service
metadata:
  name: rag-api-service
spec:
  selector:
    app: rag-api
  ports:
  - port: 80
    targetPort: 8000
  type: LoadBalancer
```

#### 3. Развертывание

```bash
kubectl apply -f deployment.yaml
kubectl apply -f service.yaml
```

## Оптимизация производительности

### 1. Кэширование с Redis

```python
# cache.py
import redis
import pickle

class EmbeddingCache:
    def __init__(self):
        self.redis = redis.Redis(host='localhost', port=6379, db=0)
    
    def get(self, text):
        key = f"emb:{hash(text)}"
        cached = self.redis.get(key)
        if cached:
            return pickle.loads(cached)
        return None
    
    def set(self, text, embedding):
        key = f"emb:{hash(text)}"
        self.redis.setex(key, 3600, pickle.dumps(embedding))
```

### 2. GPU ускорение

```bash
# Замена faiss-cpu на faiss-gpu
pip uninstall faiss-cpu
pip install faiss-gpu
```

### 3. Асинхронность

```python
# async_api.py
from fastapi import FastAPI
import asyncio

@app.post("/answer")
async def answer_question(request: QuestionRequest):
    result = await asyncio.to_thread(
        pipeline.process_question,
        question=request.question,
        answer_type=request.answer_type
    )
    return result
```

## Безопасность

### 1. API ключи

```python
# api.py
from fastapi import Security, HTTPException
from fastapi.security import APIKeyHeader

API_KEY = "your-secret-key"
api_key_header = APIKeyHeader(name="X-API-Key")

async def verify_api_key(api_key: str = Security(api_key_header)):
    if api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API key")
    return api_key

@app.post("/answer")
async def answer_question(
    request: QuestionRequest,
    api_key: str = Depends(verify_api_key)
):
    ...
```

### 2. Rate limiting

```python
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

@app.post("/answer")
@limiter.limit("10/minute")
async def answer_question(request: Request, ...):
    ...
```

## Backup и восстановление

### Backup индекса

```bash
# Создание backup
tar -czf index-backup-$(date +%Y%m%d).tar.gz index/

# Копирование в S3
aws s3 cp index-backup-*.tar.gz s3://your-bucket/backups/
```

### Восстановление

```bash
# Скачивание из S3
aws s3 cp s3://your-bucket/backups/index-backup-20240315.tar.gz .

# Распаковка
tar -xzf index-backup-20240315.tar.gz
```

## Troubleshooting

### Проблема: Out of Memory

```bash
# Увеличьте swap
sudo fallocate -l 4G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

### Проблема: Медленная работа

```bash
# Проверьте CPU/RAM
htop

# Проверьте диск
iostat -x 1

# Оптимизируйте параметры
TOP_K_RETRIEVAL=20
TOP_K_RERANK=3
```

### Проблема: API не отвечает

```bash
# Проверьте процесс
ps aux | grep python

# Проверьте порт
netstat -tulpn | grep 8000

# Проверьте логи
tail -f api.log
```

## Заключение

Система готова к развертыванию в различных окружениях:
- ✅ Локальное развертывание
- ✅ Docker контейнеры
- ✅ Облачные платформы (AWS, GCP, Azure)
- ✅ Kubernetes кластеры
- ✅ Мониторинг и масштабирование

Выберите подходящий вариант и следуйте инструкциям! 🚀
