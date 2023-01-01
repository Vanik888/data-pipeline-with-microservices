FROM python:3.9

WORKDIR /app
COPY ../main.py \
    ../pyproject.toml \
    ../poetry.lock ./

RUN pip install poetry==1.1.13
RUN poetry install

ENTRYPOINT poetry run python /app/main.py
