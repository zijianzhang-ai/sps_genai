FROM python:3.13-slim-bookworm

COPY --from=ghcr.io/astral-sh/uv:0.13.9 /uv /uvx /bin/

WORKDIR /code

COPY pyproject.toml uv.lock README.md .python-version /code/
COPY src /code/src
COPY app /code/app
COPY main.py /code/main.py
COPY helper_lib /code/helper_lib
COPY checkpoints_assignment2/best_model.pth /code/checkpoints_assignment2/best_model.pth
COPY checkpoints_rnn/rnn_model.pth /code/checkpoints_rnn/rnn_model.pth
COPY checkpoints_assignment3/assignment3_gan.pth /code/checkpoints_assignment3/assignment3_gan.pth

RUN uv sync --frozen
COPY checkpoints_llm /code/checkpoints_llm
COPY checkpoints_assignment4_diffusion/diffusion_best.pth /code/checkpoints_assignment4_diffusion/diffusion_best.pth
COPY checkpoints_assignment4_ebm/ebm_latest.pth /code/checkpoints_assignment4_ebm/ebm_latest.pth

EXPOSE 80

CMD ["/code/.venv/bin/fastapi", "run", "main.py", "--host", "0.0.0.0", "--port", "80"]