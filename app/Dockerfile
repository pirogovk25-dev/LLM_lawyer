FROM python:3.11-slim

WORKDIR /app

RUN sed -i 's/deb.debian.org/mirror.yandex.ru/g' /etc/apt/sources.list.d/debian.sources 2>/dev/null || \
    sed -i 's/deb.debian.org/mirror.yandex.ru/g' /etc/apt/sources.list 2>/dev/null || true && \
    apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8001

CMD ["uvicorn", "main:app_api", "--host", "0.0.0.0", "--port", "8001"]
