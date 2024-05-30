# Copyright (C) 2024, RTE (http://www.rte-france.com)
# Copyright (C) 2024 Savoir-faire Linux, Inc.
# SPDX-License-Identifier: Apache-2.0
FROM ubuntu:22.04

ENV DEBIAN_FRONTEND noninteractive

ENV LANG en_US.UTF-8
ENV LANGUAGE en_US.UTF-8
ENV LC_ALL en_US.UTF-8

RUN apt-get update && apt-get install -y --no-install-recommends \
    bash \
    bc \
    bison \
    ca-certificates \
    wget

RUN wget https://github.com/bpftrace/bpftrace/releases/download/v0.20.4/bpftrace
RUN chmod +x bpftrace
