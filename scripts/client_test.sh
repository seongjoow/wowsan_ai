#!/bin/bash

echo "Running gRPC client test via Docker..."
docker run --rm $CLIENT_IMAGE

EXIT_CODE=$?
if [ $EXIT_CODE -eq 0 ]; then
    echo "Client test executed successfully!"
else
    echo "Client test failed with exit code $EXIT_CODE!"
fi

exit $EXIT_CODE
