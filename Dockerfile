FROM ubuntu:20.04

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
RUN pip3 install git+https://github.com/minerllabs/minerl@v1.0.0

COPY . .

CMD ["/bin/sh", "xvfb-run python3 -u ./src/minerl3161/main.py"]
