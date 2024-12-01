@echo off

set CLIENT_IMAGE=grpc-client

docker build -f Dockerfile.client -t %CLIENT_IMAGE% .
if %ERRORLEVEL% NEQ 0 (
    echo Error: Failed to build client Docker image!
    exit /b 1
)

echo Running gRPC client test via Docker...
docker run --rm %CLIENT_IMAGE%

if %ERRORLEVEL% EQU 0 (
    echo Client test executed successfully!
) else (
    echo Client test failed with exit code %ERRORLEVEL%!
)

exit /b %ERRORLEVEL%
