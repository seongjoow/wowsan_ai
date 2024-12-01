# grpc-server.dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install required packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt


# The actual python files will be mounted as volume
# VOLUME ["/app/grpc_src"]

CMD ["python", "/app/grpc_src/server.py"]