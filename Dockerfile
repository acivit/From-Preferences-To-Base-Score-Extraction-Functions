FROM ubuntu:22.04

# Install dependencies 
RUN apt-get update && \
    apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    git \
    curl \
    build-essential \
    libssl-dev \
    libffi-dev \
    python3-dev
# Create a virtual environment
RUN python3 -m venv /venv
# Activate the virtual environment
ENV PATH="/venv/bin:$PATH"
# Install Python packages
RUN pip install --upgrade pip

#Copy files
COPY requirements.txt /app/requirements.txt
COPY . /app
WORKDIR /app
# Install Python dependencies
RUN pip install -r requirements.txt
#RUN git submodule update --init --recursive --remote
WORKDIR /app/Quantitative-Bipolar-Argumentation
RUN pip install -e .
WORKDIR /app

