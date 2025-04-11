@echo off

set CLIENT_IMAGE=grpc-client

echo Running gRPC client test via Docker...
docker run --rm %CLIENT_IMAGE%

if %ERRORLEVEL% EQU 0 (
    echo Client test executed successfully!
) else (
    echo Client test failed with exit code %ERRORLEVEL%!
)

exit /b %ERRORLEVEL%
