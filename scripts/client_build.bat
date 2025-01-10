@echo off

set CLIENT_IMAGE=grpc-client

docker build -f Dockerfile.client -t %CLIENT_IMAGE% .
if %ERRORLEVEL% NEQ 0 (
    echo Error: Failed to build client Docker image!
    exit /b 1
)
