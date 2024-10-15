FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime
RUN apt-get update && \
      apt-get -y install sudo
RUN useradd -rm -d /home/ubuntu -s /bin/bash -g root -G sudo -u 1010 ubuntu
USER ubuntu
WORKDIR /home/ubuntu