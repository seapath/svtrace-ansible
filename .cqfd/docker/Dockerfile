# Copyright (C) 2021, RTE (http://www.rte-france.com)
# SPDX-License-Identifier: Apache-2.0

FROM ubuntu:24.04
RUN userdel -r ubuntu

ENV DEBIAN_FRONTEND noninteractive

ENV LANG en_US.UTF-8
ENV LANGUAGE en_US.UTF-8
ENV LC_ALL en_US.UTF-8

RUN apt-get update && apt-get install -y --no-install-recommends \
	ansible \
	bash \
	locales \
	python3 \
	python3-pip \
	python3-matplotlib \
	python3-numpy \
	rsync \
	yamllint


RUN sed -i "s/# en_US\.UTF-8 UTF-8/en_US\.UTF-8 UTF-8/" /etc/locale.gen
RUN locale-gen
RUN dpkg-reconfigure locales
ADD check_yaml.sh /usr/bin/check_yaml
