# pull official base image
FROM python:3.12-slim

# set work directory
WORKDIR /usr/src/app

ARG REQUIREMENTS_FILE=requirements/base.txt

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        default-jdk-headless \
        libgl1 \
        libglib2.0-0 \
        netcat-openbsd \
        curl \
    && rm -rf /var/lib/apt/lists/*

# copy uv from official image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# install python dependencies
COPY ./requirements ./requirements
RUN uv pip install --system --no-cache -r "${REQUIREMENTS_FILE}"

# copy docker-entrypoint.sh
COPY ./docker-entrypoint.sh ./docker-entrypoint.sh

# copy project
COPY . .

RUN ["chmod", "+x", "./docker-entrypoint.sh"]
