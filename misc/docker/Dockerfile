# start from base image
FROM ubuntu:22.04

# update system packages and install prerequisites
RUN apt-get update && apt-get install -y \
    software-properties-common \
    wget \
    curl \
    build-essential \
    libssl-dev \
    zlib1g-dev \
    libncurses5-dev \
    libsqlite3-dev \
    libreadline-dev \
    libffi-dev \
    libbz2-dev \
    liblzma-dev

# set timezone information to bypass tzdata prompt
ENV TZ=Etc/UTC
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# add the deadsnakes PPA for Python 3.8
RUN add-apt-repository ppa:deadsnakes/ppa -y && \
    apt-get update

# install Python 3.8 and pip
RUN apt-get install -y python3.8 python3.8-venv python3.8-dev python3-pip

# set Python 3.8 as the default version
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 1 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.8 1

# copy in project files
COPY . /opt/app

# add to path
ENV PYTHONPATH="${PYTHONPATH}:/opt/app"

# set up a working directory
WORKDIR /opt/app

# install project dependencies
RUN pip install -r /opt/app/docker/requirements.txt

# run application
ENTRYPOINT python -u /opt/app/BGL/data_process.py




