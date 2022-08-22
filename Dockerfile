FROM nvidia/cuda:11.0.3-cudnn8-runtime-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive

# openjdk8
RUN apt update -y && \
    apt install -y software-properties-common && \
    add-apt-repository -y ppa:openjdk-r/ppa && \
    apt update -y && \
    apt install -y openjdk-8-jdk && \
    rm -rf /var/lib/apt/lists/* && \
    update-alternatives --set java /usr/lib/jvm/java-8-openjdk-amd64/jre/bin/java

# xvfb
RUN apt update -y && \
    apt install -y xvfb && \
    rm -rf /var/lib/apt/lists/*

# git
RUN apt update -y && \
    apt install -y git

# python
RUN apt install -y python3 && \
    apt update -y && \
    apt install -y pip && \
    rm -rf /var/lib/apt/lists/*

# MineRL
RUN pip3 install pyyaml
RUN pip3 install minerl==0.4.4

# Requirements
COPY requirements.txt .
RUN pip3 install -r requirements.txt

COPY . .

ENV PYTHONPATH "${PYTHONPATH}:/src"
WORKDIR /tests/

# run tests
RUN xvfb-run pytest

CMD ["/bin/sh", "-c", "xvfb-run python3 -u /src/minerl3161/main.py"]
