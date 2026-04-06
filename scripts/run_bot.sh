#!/bin/bash
# scripts/run_bot.sh – Start Rasa server + action server
set -e

RASA_MODEL_DIR="./rasa_bot/models"
RASA_PORT=5005
ACTION_PORT=5055

echo "=== PharmaAI Rasa Launcher ==="

# Start action server in background
echo "[1/2] Starting Rasa Action Server on port $ACTION_PORT..."
python -m rasa_sdk \
  --actions rasa_bot.actions \
  --port "$ACTION_PORT" \
  --debug &
ACTION_PID=$!

sleep 3

# Train model if needed
if [ ! -d "$RASA_MODEL_DIR" ] || [ -z "$(ls -A $RASA_MODEL_DIR 2>/dev/null)" ]; then
  echo "[!] No Rasa model found. Training..."
  rasa train --domain rasa_bot/domain.yml \
             --data rasa_bot/nlu.yml rasa_bot/stories.yml rasa_bot/rules.yml \
             --config rasa_bot/config.yml \
             --out "$RASA_MODEL_DIR"
fi

# Start Rasa server
echo "[2/2] Starting Rasa Server on port $RASA_PORT..."
rasa run \
  --enable-api \
  --cors "*" \
  --port "$RASA_PORT" \
  --model "$RASA_MODEL_DIR" \
  --endpoints rasa_bot/endpoints.yml \
  --debug &
RASA_PID=$!

echo "Rasa running (PID=$RASA_PID), Action server (PID=$ACTION_PID)"
echo "API: http://localhost:$RASA_PORT"

# Wait
wait $RASA_PID