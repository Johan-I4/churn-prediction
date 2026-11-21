FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Opcional: docker build --build-arg WITH_LLM=1 ... instala el SDK de Anthropic.
ARG WITH_LLM=0
COPY requirements-llm.txt .
RUN if [ "$WITH_LLM" = "1" ]; then pip install --no-cache-dir -r requirements-llm.txt; fi

COPY src ./src
COPY prompts ./prompts
COPY docs ./docs
COPY data ./data

# Entrena dentro de la imagen para que el modelo siempre coincida
# con la versión de scikit-learn instalada en el contenedor.
RUN python -m src.train --data data/WA_Fn-UseC_-Telco-Customer-Churn.csv

EXPOSE 8000
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
