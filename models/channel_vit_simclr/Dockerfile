FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04

WORKDIR /app

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl ca-certificates unzip && \
    rm -rf /var/lib/apt/lists/*

RUN curl -fsSL https://pixi.sh/install.sh | sh

ENV PATH="/root/.pixi/bin:${PATH}"

RUN chmod -R 777 /root

COPY pixi.toml pixi.lock ./

RUN CONDA_OVERRIDE_CUDA="12.8" pixi install --locked 

RUN pixi shell-hook --shell bash  > /etc/profile.d/pixi.sh

CMD ["/bin/bash"]