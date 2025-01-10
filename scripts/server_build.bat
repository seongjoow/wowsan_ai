@echo off

if not exist protos mkdir protos
if not exist src mkdir src
if not exist generated mkdir generated

if not exist "protos\embedding_service.proto" (
    echo Error: protos\embedding_service.proto not found!
    exit /b 1
)

if not exist "generated\embedding_service_pb2.py" (
    echo Error: gRPC code generation failed! Missing embedding_service_pb2.py
    exit /b 1
)

if not exist "generated\embedding_service_pb2_grpc.py" (
    echo Error: gRPC code generation failed! Missing embedding_service_pb2_grpc.py
    exit /b 1
)

echo Building Docker images...
docker-compose build
if %ERRORLEVEL% NEQ 0 (
    echo Error: Failed to build Docker images!
    exit /b 1
)
