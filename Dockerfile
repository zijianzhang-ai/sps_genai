FROM python:3.13-slim-bookworm

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

ADD https://astral.sh/uv/install.sh /uv-installer.sh

RUN sh /uv-installer.sh \
    && rm /uv-installer.sh

ENV PATH="/root/.local/bin:$PATH"

WORKDIR /code

COPY pyproject.toml uv.lock README.md .python-version /code/
COPY src /code/src
COPY app /code/app
COPY main.py /code/main.py
COPY helper_lib /code/helper_lib
COPY checkpoints_assignment2/best_model.pth /code/checkpoints_assignment2/best_model.pth

RUN uv sync --frozen

EXPOSE 80

CMD ["/code/.venv/bin/fastapi", "run", "main.py", "--host", "0.0.0.0", "--port", "80"]