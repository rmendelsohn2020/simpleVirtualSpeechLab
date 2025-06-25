#!/bin/bash

# Overnight particle swarm optimization launcher
# This script starts the optimization in the background and saves all output

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Create logs directory if it doesn't exist
mkdir -p logs

# Generate timestamp for this run
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Set up log files
LOG_FILE="logs/overnight_run_${TIMESTAMP}.log"
ERROR_LOG="logs/overnight_run_${TIMESTAMP}_error.log"

echo "Starting overnight particle swarm optimization..."
echo "Timestamp: ${TIMESTAMP}"
echo "Log file: ${LOG_FILE}"
echo "Error log: ${ERROR_LOG}"
echo ""

# Start the optimization in the background with nohup
# This ensures it continues running even if the SSH connection drops
nohup python src/run_overnight_optimization.py > "${LOG_FILE}" 2> "${ERROR_LOG}" &

# Get the process ID
PID=$!

echo "Optimization started with PID: ${PID}"
echo "You can monitor progress using:"
echo "  python src/utils/monitor_optimization.py --watch"
echo ""
echo "To check if it's still running:"
echo "  ps aux | grep ${PID}"
echo ""
echo "To view the log in real-time:"
echo "  tail -f ${LOG_FILE}"
echo ""
echo "To view error log:"
echo "  tail -f ${ERROR_LOG}"
echo ""
echo "To stop the optimization:"
echo "  kill ${PID}"
echo ""

# Save PID to file for easy reference
echo "${PID}" > "logs/overnight_run_${TIMESTAMP}.pid"

echo "PID saved to: logs/overnight_run_${TIMESTAMP}.pid"
echo "Optimization is now running in the background." 