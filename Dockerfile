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

RUN conda create -n inspector_env python=3.11 -y \
    && conda run -n inspector_env pip install --no-cache-dir -r requirements.txt

ENV PATH="/opt/conda/envs/inspector_env/bin:$PATH"
ENV ENVIRONMENT=prod

COPY . .
RUN find . -name '*.db' -delete

CMD ["python", "bot.py"]