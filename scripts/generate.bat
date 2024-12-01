@echo off
setlocal enabledelayedexpansion

if not exist protos mkdir protos
if not exist generated mkdir generated

echo Proto files in 'protos' directory:
dir /b protos

echo Generating gRPC code...

:: build proto doc
set "PROTO_FILES="
for %%F in (protos\*.proto) do (
    set "PROTO_FILES=!PROTO_FILES! /app/%%~nxF"
)

:: Run docker container, build
docker run --rm -v "%cd%\protos:/app/protos" ^
                -v "%cd%\generated:/app/generated" ^
                %docker build -q -f Dockerfile.proto .% ^
                python -m grpc_tools.protoc ^
                    -I/app/protos ^
                    --python_out=/app/generated ^
                    --grpc_python_out=/app/generated ^
                    !PROTO_FILES!

type nul > generated\__init__.py

:: Check generated result
if not exist "generated\embedding_service_pb2.py" (
    echo Error: gRPC code generation failed!
    exit /b 1
)

if not exist "generated\embedding_service_pb2_grpc.py" (
    echo Error: gRPC code generation failed!
    exit /b 1
)

echo gRPC code generation completed successfully!
exit /b 0
