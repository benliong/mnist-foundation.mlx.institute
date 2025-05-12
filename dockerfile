# Streamlit + PyTorch
# multi-arch, tiny base  [oai_citation:0‡Docker Hub](https://hub.docker.com/layers/library/python/3.11-slim/images/sha256-1591aa8c01b5b37ab31dbe5662c5bdcf40c2f1bce4ef1c1fd24802dae3d01052?context=explore&utm_source=chatgpt.com)
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Streamlit listens on 8501
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]