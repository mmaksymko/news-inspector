FROM python:3.11-bookworm

RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    gfortran \
    libc6 \
    && wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /miniconda.sh \
    && bash /miniconda.sh -b -p /opt/conda \
    && rm /miniconda.sh \
    && apt-get clean && rm -rf /var/lib/apt/lists/*


ENV PATH="/opt/conda/bin:$PATH"

WORKDIR /app

COPY requirements.txt .

RUN conda create -n app_env python=3.11 -y \
    && conda run -n app_env pip install --no-cache-dir -r requirements.txt

ENV PATH="/opt/conda/envs/app_env/bin:$PATH"

COPY . .

CMD ["python", "bot.py"]