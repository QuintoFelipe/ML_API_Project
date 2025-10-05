FROM python:3.11-slim

# 1. Instalar dependencias del sistema necesarias para CatBoost (libgomp1)
RUN apt-get update && \
    apt-get install -y --no-install-recommends libgomp1 && \
    rm -rf /var/lib/apt/lists/*

# 2. Directorio de trabajo
WORKDIR /app

# 3. Copiar requirements y instalarlos
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Copiar el código de la aplicación
COPY . .

# 5. Exponer el puerto 8000
EXPOSE 8000

# 6. Comando por defecto para arrancar Uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]