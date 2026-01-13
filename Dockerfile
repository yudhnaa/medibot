# pull official base image
FROM python:3.13-alpine

# set work directory
WORKDIR /usr/src/app

# copy uv from official image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# install python dependencies
COPY ./requirements ./requirements
RUN uv pip install --system --no-cache -r requirements/dev.txt

# copy docker-entrypoint.sh
COPY ./docker-entrypoint.sh ./docker-entrypoint.sh

# copy project
COPY . .

RUN ["chmod", "+x", "./docker-entrypoint.sh"]
