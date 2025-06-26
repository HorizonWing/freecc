FROM python:3.10.17

# 安装 uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh && \
    echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc && \
    export PATH="$HOME/.local/bin:$PATH" && \
    uv --version

# 创建 freecc 命令的脚本
RUN echo '#!/bin/bash\ncd /code/freecc\nuv run uvicorn server:app --host 0.0.0.0 --port 8082 --reload' > /usr/local/bin/freecc && \
    chmod +x /usr/local/bin/freecc

COPY . /code/freecc

EXPOSE 8082

WORKDIR /code

CMD ["freecc"]