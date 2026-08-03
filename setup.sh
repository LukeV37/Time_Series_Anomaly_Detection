#!/bin/bash

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
SCRIPT_PATH="${BASH_SOURCE[0]:-$0}"
REPO_ROOT="$(cd -- "$(dirname -- "$SCRIPT_PATH")" && pwd)"

echo -e "${YELLOW}Preparing root repo environment...${NC}"

if [ -f "$REPO_ROOT/export.sh" ]; then
  echo -e "${YELLOW}Sourcing $REPO_ROOT/export.sh${NC}"
  source "$REPO_ROOT/export.sh"
fi

if [ -n "$VENV_DIR" ]; then
  ENV_DIR="$VENV_DIR/time_series_anomaly_detection"
else
  ENV_DIR="$REPO_ROOT/venv"
fi

echo -e "${YELLOW}Using virtual environment: $ENV_DIR${NC}"

if [ ! -f "$ENV_DIR/bin/activate" ]; then
  echo -e "${YELLOW}Environment not found. Creating a new virtual environment...${NC}"
  mkdir -p "$(dirname -- "$ENV_DIR")"
  python3 -m venv "$ENV_DIR"
  source "$ENV_DIR/bin/activate"
  echo -e "${YELLOW}Upgrading pip...${NC}"
  pip install --upgrade pip
  echo -e "${YELLOW}Installing requirements from $REPO_ROOT/requirements.txt...${NC}"
  pip install -r "$REPO_ROOT/requirements.txt"
else
  echo -e "${YELLOW}Environment already exists. Activating...${NC}"
  source "$ENV_DIR/bin/activate"
fi

echo -e "${GREEN}Environment ready.${NC}"
