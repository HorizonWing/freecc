FROM astral/uv:python3.10-alpine

COPY . /code/freecc

EXPOSE 8082

WORKDIR /code
RUN cd freecc && uv sync

CMD ["uv", "run", "--directory", "freecc", "uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8082"]