#!/usr/bin/env bash
set -euo pipefail

# ──────────────────────────────
# CONFIG
# ──────────────────────────────
APP_DIR="/var/www/seedsoftruth"
VENV_DIR="$APP_DIR/venv"
SERVICE_NAME="seedsoftruth"
BRANCH="main"

echo "=== Deploy started at $(date) ==="

# ──────────────────────────────
# SAFETY CHECKS
# ──────────────────────────────
if [[ ! -d "$APP_DIR/.git" ]]; then
  echo "ERROR: $APP_DIR is not a git repository"
  exit 1
fi

if [[ ! -d "$VENV_DIR" ]]; then
  echo "ERROR: virtualenv not found at $VENV_DIR"
  exit 1
fi

# ──────────────────────────────
# UPDATE CODE
# ──────────────────────────────
cd "$APP_DIR"

echo "Fetching latest code..."
git fetch origin

echo "Resetting to origin/$BRANCH..."
git reset --hard "origin/$BRANCH"

# Optional: verify clean tree
if [[ -n "$(git status --porcelain)" ]]; then
  echo "ERROR: Working tree is dirty after reset"
  exit 1
fi

# Must run under seeddev for proper permissions
chown -R seeddev:seeddev "$APP_DIR"

# ──────────────────────────────
# UPDATE DEPENDENCIES
# ──────────────────────────────
echo "Activating virtualenv..."
source "$VENV_DIR/bin/activate"

echo "Upgrading pip..."
pip install --upgrade pip

echo "Installing requirements..."
pip install -r requirements.txt

# ──────────────────────────────
# RESTART SERVICE
# ──────────────────────────────
echo "Restarting systemd service..."
sudo systemctl restart "$SERVICE_NAME"

echo "Waiting for service health..."
sleep 2
systemctl is-active --quiet "$SERVICE_NAME" || {
  echo "ERROR: Service failed to start"
  journalctl -u "$SERVICE_NAME" -n 50
  exit 1
}

echo "=== Deploy finished successfully at $(date) ==="
