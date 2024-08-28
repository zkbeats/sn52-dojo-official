FROM python:3.11-slim-buster

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
    build-essential gcc curl git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

COPY . .

RUN pip install --upgrade pip && pip install -e .

COPY entrypoints.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/entrypoints.sh

ENTRYPOINT ["entrypoints.sh"]
