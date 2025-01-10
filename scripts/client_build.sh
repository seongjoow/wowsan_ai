#!/bin/bash
CLIENT_IMAGE="grpc-client"
docker build -f Dockerfile.client -t $CLIENT_IMAGE .
if [ $? -ne 0 ]; then
    echo "Error: Failed to build client Docker image!"
    exit 1
fi
