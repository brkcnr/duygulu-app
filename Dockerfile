# Base image
FROM python:3.10-slim-bookworm

# Ortam değişkenleri
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

# Sistem bağımlılıklarını yükle ve güvenlik temizliği yap
RUN apt-get update && apt-get upgrade -y && \
    apt-get install -y --no-install-recommends gcc libpq-dev postgresql-client && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Çalışma dizini
WORKDIR /app

# Gereken dosyaları kopyala ve bağımlılıkları yükle
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Uygulama ve model dosyaları
COPY . .
COPY results/checkpoint-80000/ /app/results/checkpoint-80000/

# wait-for-it.sh
COPY wait-for-it.sh /wait-for-it.sh
RUN chmod +x /wait-for-it.sh

# Başlatma komutu—db hazır olana kadar bekle
ENTRYPOINT ["/wait-for-it.sh", "db", "--"]
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
