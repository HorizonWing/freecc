FROM python:3.10.17

# 安装 uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh && \
    echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc && \
    export PATH="$HOME/.local/bin:$PATH" && \
    uv --version

# 创建 freecc 命令的脚本
RUN echo '#!/bin/bash\ncd /code/freecc\nuv run uvicorn server:app --host 0.0.0.0 --port 8082 --reload' > /usr/local/bin/freecc && \
    chmod +x /usr/local/bin/freecc && \
    echo '#!/bin/bash\nexport ANTHROPIC_BASE_URL=http://localhost:8082 && claude' > /usr/local/bin/cc && \
    chmod +x /usr/local/bin/cc

COPY . /code/freecc

WORKDIR /code

CMD ["freecc"]