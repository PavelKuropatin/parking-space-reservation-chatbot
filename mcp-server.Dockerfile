FROM python:3.13-slim

WORKDIR /app

RUN pip install --no-cache-dir "fastmcp[server]==3.4.2"

COPY chatbot/logging.py ./chatbot/logging.py
COPY chatbot/__init__.py ./chatbot/__init__.py
COPY chatbot/mcp/ ./chatbot/mcp/

EXPOSE 8088

CMD ["python", "-m", "chatbot.mcp.mcp_server"]
