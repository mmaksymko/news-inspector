# Use an official Python runtime as a parent image
FROM python:3.11-slim-bookworm

# Install Miniconda and gfortran
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    gfortran \
    && wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /miniconda.sh \
    && bash /miniconda.sh -b -p /opt/conda \
    && rm /miniconda.sh \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Add Conda to PATH
ENV PATH="/opt/conda/bin:$PATH"

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file
COPY requirements.txt .

# Create a Conda environment and install dependencies
RUN conda create -n app_env python=3.11 -y \
    && conda run -n app_env pip install --no-cache-dir -r requirements.txt

# Activate the Conda environment
ENV PATH="/opt/conda/envs/app_env/bin:$PATH"

# Copy the rest of the application code
COPY . .

# Expose the port your app runs on (if applicable)
EXPOSE 8080

# Set the command to run your application
CMD ["python", "bot.py"]