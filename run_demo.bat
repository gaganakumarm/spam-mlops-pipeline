@echo off
echo  Starting Automated MLOps Architecture...

:: 1. Start Docker containers in the background
docker-compose up -d --build

echo  Waiting for services to stabilize...
timeout /t 10

:: 2. Automatically open the Dashboards
echo  Opening MLflow and Swagger UI...
start http://localhost:5000
start http://localhost:8000/docs

echo  Architecture is LIVE. 
echo  Go to the Swagger browser window to test input!
pause   