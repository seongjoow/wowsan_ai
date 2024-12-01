#!/bin/bash
mkdir -p protos generated


echo "Proto files in 'protos' directory:"
ls protos

echo "Generating gRPC code..."

PROTO_FILES=$(ls protos/*.proto | xargs -I{} echo /app/{})
docker run --rm -v "$(pwd)/protos:/app/protos" \
                -v "$(pwd)/generated:/app/generated" \
                $(docker build -q -f Dockerfile.proto .) \
                python -m grpc_tools.protoc \
                    -I/app/protos \
                    --python_out=/app/generated \
                    --grpc_python_out=/app/generated \
                    --mypy_out=/app/generated \
                    $PROTO_FILES

touch generated/__init__.py

if [ ! -f "generated/embedding_service_pb2.py" ] || [ ! -f "generated/embedding_service_pb2_grpc.py" ]; then
    echo "Error: gRPC code generation failed!"
    exit 1
fi

echo "gRPC code generation completed successfully!"
