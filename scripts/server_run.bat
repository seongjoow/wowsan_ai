
echo Starting gRPC server...
docker-compose up grpc-server
if %ERRORLEVEL% NEQ 0 (
    echo Error: Failed to start gRPC server!
    exit /b 1
)

echo Setup completed successfully!
exit /b 0
