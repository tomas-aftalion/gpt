# Build argument to select cpu or gpu base
ARG BASE=cpu

FROM ubuntu:24.04 AS cpu-base
FROM nvidia/cuda:11.8.0-base-ubuntu22.04 AS gpu-base
FROM ${BASE}-base AS base

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Set PIP environment variables
ENV PIP_BREAK_SYSTEM_PACKAGES=1
ENV PIP_NO_CACHE_DIR=1

COPY requirements.txt .
RUN pip3 install -r requirements.txt

# Copy all code from repository
COPY . .

RUN pip3 install -e .
