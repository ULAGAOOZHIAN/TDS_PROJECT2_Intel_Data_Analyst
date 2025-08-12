# ---- Base ----
FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MPLCONFIGDIR=/app/.matplotlib_tmp \
    DUCKDB_EXTENSION_DIRECTORY=/app/.duckdb_extensions

WORKDIR /app

# System deps: curl (debug), and fonts for matplotlib if needed later
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    fonts-dejavu-core \
 && rm -rf /var/lib/apt/lists/*

# Create writable caches (match your code)
RUN mkdir -p /app/.matplotlib_tmp /app/.duckdb_extensions /app/logs

# ---- Python deps ----
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ---- Preinstall DuckDB extensions at build-time ----
# This downloads & caches extensions into DUCKDB_EXTENSION_DIRECTORY inside the image
RUN python - <<'PY'
import duckdb, os
os.makedirs('/app/.duckdb_extensions', exist_ok=True)
con = duckdb.connect()
# Use the official repo (default), then install & load common extensions you use
for ext in ['httpfs','parquet','json','spatial']:
    try:
        con.execute(f"INSTALL {ext}; LOAD {ext};")
        print("Installed:", ext)
    except Exception as e:
        print("WARNING installing", ext, "->", e)
con.close()
PY

# ---- App ----
COPY . .

# Hugging Face Spaces exposes 7860
EXPOSE 7860

# Start FastAPI via Uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
