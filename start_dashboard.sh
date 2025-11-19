#!/bin/bash

# Function to handle cleanup on exit
cleanup() {
    echo "Stopping servers..."
    kill $(jobs -p) 2>/dev/null
    exit
}

trap cleanup SIGINT SIGTERM

echo "Starting Dashboard Backend on port 8001..."
uv run python dashboard/server.py &
BACKEND_PID=$!

echo "Starting Dashboard Frontend..."
cd dashboard/frontend
npm run dev -- --port 5173 &
FRONTEND_PID=$!

echo "Dashboard running!"
echo "Backend: http://localhost:8001"
echo "Frontend: http://localhost:5173"
echo "Press Ctrl+C to stop."

wait
