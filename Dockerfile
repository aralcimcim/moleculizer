FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN useradd --create-home --uid 10001 app && chown app:app /app
COPY --chown=app:app build_index.py evidence.py datasets.py ./
USER app
ARG INDEX_LIMIT=100
RUN python build_index.py --dataset all --limit "$INDEX_LIMIT"
COPY --chown=app:app app.py rag.py ui.py styles.css ./
COPY --chown=app:app .streamlit .streamlit
USER root
RUN apt-get update && apt-get install -y --no-install-recommends libxrender1 libxext6 libsm6 libexpat1 && rm -rf /var/lib/apt/lists/*
USER app
RUN python -c "from rdkit import Chem; from rdkit.Chem import Draw; Draw.MolToImage(Chem.MolFromSmiles('CCO'))"
EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health', timeout=4)"
CMD ["python", "-m", "streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true"]
