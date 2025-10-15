
FROM python:3.12.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    build-essential \
    && rm -rf /var/lib/apt/lists/*


ADD https://astral.sh/uv/install.sh /uv-installer.sh
RUN sh /uv-installer.sh && rm /uv-installer.sh
ENV PATH="/root/.local/bin/:$PATH"

WORKDIR /app

COPY pyproject.toml uv.lock* ./
RUN uv venv --python 3.12.11 && uv sync

RUN echo '#!/bin/bash\nuv run streamlit run app.py --server.headless=true --server.port 8086 --server.address 0.0.0.0 --server.fileWatcherType none' > /app/start.sh && chmod +x /app/start.sh

COPY . .

EXPOSE 8086

CMD ["/app/start.sh"]