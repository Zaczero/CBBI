FROM ubuntu:focal

ENV DEBIAN_FRONTEND=noninteractive
ENV SHELL=/bin/bash

RUN apt update
RUN apt install -yq software-properties-common
RUN add-apt-repository -y ppa:deadsnakes/ppa
RUN apt install -yq python3.9 pipenv

RUN mkdir -p /cbbi
COPY . /cbbi

WORKDIR /cbbi
RUN pipenv install

ENTRYPOINT [ "pipenv" ] 
CMD [ "run", "python", "main.py" ]