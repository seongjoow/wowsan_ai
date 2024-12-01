#!/bin/bash
proto_path="protos"
proto_name="embedding_service"

generated_path="generated"
pb_name="${generated_path}/${proto_name}_pb2.py"
pb_grpc_name="${generated_path}/${proto_name}_pb2_grpc.py"

mkdir -p protos src generated

if [ ! -f "$proto_path"/"$proto_name".proto ]; then
    echo "Error: "$proto_path"/"$proto_name".proto not found!"
    exit 1
fi

if [ ! -f "$pb_name" ] || [ ! -f "$pb_grpc_name" ]; then
    echo "Error: gRPC code generation failed!"
    exit 1
fi

echo "Building Docker images..."
docker compose build
