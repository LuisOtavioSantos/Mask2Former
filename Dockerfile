FROM ubuntu:22.04
WORKDIR /lapix
ENV DEBIAN_FRONTEND=noninteractive
    
    
# Dependências iniciais
RUN apt-get update && \
    apt-get --no-install-recommends install -yq \
        curl \
        g++ \
        gcc \
        git \
        libgeos-dev \
        libldap2-dev \
        libsasl2-dev \
        make \
        nasm \
        pkg-config \
        python3-pip \
        python3.10-venv \
        ffmpeg \
        bzip2 \
        ca-certificates \
        libgl1 \
        libgomp1 \
        tzdata \
        unrar \
        libsm6 \
        libxext6 \
        gnupg \
        wget \
        software-properties-common \
        build-essential \
        libssl-dev \
        zlib1g-dev \
        libncurses5-dev \
        libsqlite3-dev \
        libreadline-dev \
        libffi-dev \
        libbz2-dev \
        liblzma-dev \
    && rm -rf /var/lib/apt/lists/*

# Instalar Python 3.11.10 a partir do código-fonte
RUN wget https://www.python.org/ftp/python/3.11.10/Python-3.11.10.tgz && \
    tar -xf Python-3.11.10.tgz && \
    cd Python-3.11.10 && \
    ./configure --enable-optimizations && \
    make -j$(nproc) && \
    make altinstall && \
    ln -s /usr/local/bin/python3.11 /usr/local/bin/python && \
    cd .. && \
    rm -rf Python-3.11.10.tgz Python-3.11.10
    
# Remover chave antiga do NVIDIA e instalar CUDA
RUN apt-key del 7fa2af80
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb && \
    dpkg -i cuda-keyring_1.1-1_all.deb && \
    apt-get update && \
    apt-get -y install cuda-toolkit-12-4

# Configurações de ambiente CUDA
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV PATH $PATH:/usr/local/cuda-12.4/bin
ENV CUDADIR /usr/local/cuda-12.4
ENV CUDA_HOME /usr/local/cuda-12.4
# ENV LD_LIBRARY_PATH $LD_LIBRARY_PATH:/usr/local/cuda-12.4/lib64
ENV LD_LIBRARY_PATH ${LD_LIBRARY_PATH}:/usr/local/cuda-12.4/lib64
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility
ENV NVIDIA_VISIBLE_DEVICES all

# Atualizar pip e instalar pacotes Python com versões específicas
RUN python -m pip install --upgrade pip==23.2.1
RUN python -m pip install -U pip setuptools==68.2.2 wheel==0.41.2
RUN python -m pip install --ignore-installed pyreadr==0.5.2 plotnine==0.14.0 Flask==3.0.3 neptune==1.13.0 wandb==0.18.5 beautifulsoup4==4.12.3 opencv-python==4.10.0.84 pandas==2.2.3 gdown==5.2.0
RUN python -m pip install --ignore-installed seaborn==0.13.2 matplotlib==3.9.2 chart_studio==1.1.0 PyYAML==6.0.2 pillow==11.0.0 google-auth==2.35.0 pytz==2024.2
RUN python -m pip install --ignore-installed plotly==5.24.1
RUN python -m pip install --ignore-installed xgboost==2.1.2 torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 pycocotools==2.0.8 transformers==4.36.2 lapixdl==0.11.0
RUN python -m pip install --ignore-installed scipy==1.14.1 scikit-learn==1.5.2 scikit-image==0.24.0 imutils==0.5.4 einops==0.8.0 timm==1.0.11 albumentations==1.4.21 ultralytics==8.3.27
RUN python -m pip install --ignore-installed kaggle==1.6.17 grad-cam==1.5.4 roboflow==1.1.48
RUN python -m pip install --ignore-installed keras==3.6.0 tensorboard==2.18.0 nltk==3.9.1
RUN python -m pip install --ignore-installed gensim==4.3.3
# RUN python -m pip install --ignore-installed -U spacy[cuda12x,transformers,lookups
# RUN python -m spacy download en_core_web_sm
# RUN python -m spacy download pt_core_news_sm
# RUN python -m spacy download en_core_web_trf
# RUN python -m spacy download pt_core_news_lg
RUN python -m pip install --ignore-installed minisom==2.3.3 susi==1.4.2
RUN python -m pip install --ignore-installed 'git+https://github.com/facebookresearch/detectron2.git@8d85329aed8506ea3672e3e208971345973ea761'
RUN python -m pip install --ignore-installed 'git+https://github.com/facebookresearch/detectron2.git@8d85329aed8506ea3672e3e208971345973ea761#subdirectory=projects/DensePose'
RUN python -m pip install --ignore-installed fastai==2.7.18
RUN python -m pip install --ignore-installed jupyterlab==4.2.5
RUN python -m pip install --ignore-installed jupyter==1.1.1
RUN python -m pip install --ignore-installed ipywidgets==8.1.5

RUN git clone https://github.com/facebookresearch/Mask2Former.git && \
    cd Mask2Former && \
    git checkout 9b0651c6c1d5b3af2e6da0589b719c514ec0d69a

ENV FORCE_CUDA 1
ENV TORCH_CUDA_ARCH_LIST "7.5;8.0;8.6+PTX"

RUN pip install -r Mask2Former/requirements.txt
RUN python Mask2Former/mask2former/modeling/pixel_decoder/ops/setup.py build develop

ENV PYTHONPATH="${PYTHONPATH}:/lapix/Mask2Former"

# Add Tini. Tini operates as a process subreaper for jupyter. This prevents
# kernel crashes.
ENV TINI_VERSION v0.6.0
ADD https://github.com/krallin/tini/releases/download/${TINI_VERSION}/tini /usr/bin/tini
RUN chmod +x /usr/bin/tini
ENTRYPOINT ["/usr/bin/tini", "--"]
EXPOSE 8888
RUN apt-get install -yq unzip
RUN python -m pip cache purge
RUN rm -rf /var/lib/apt/lists/*
RUN apt-get clean
CMD ["jupyter", "lab", "--no-browser", "--ServerApp.token='mask2former'", "--ip=0.0.0.0", "--allow-root"]

