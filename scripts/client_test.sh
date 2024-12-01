#!/bin/bash
CLIENT_IMAGE="grpc-client"
docker build -f Dockerfile.client -t $CLIENT_IMAGE .
if [ $? -ne 0 ]; then
    echo "Error: Failed to build client Docker image!"
    exit 1
fi

echo "Running gRPC client test via Docker..."
docker run --rm $CLIENT_IMAGE

EXIT_CODE=$?
if [ $EXIT_CODE -eq 0 ]; then
    echo "Client test executed successfully!"
else
    echo "Client test failed with exit code $EXIT_CODE!"
fi

exit $EXIT_CODE
