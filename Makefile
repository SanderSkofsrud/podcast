# Makefile for Podcast AdBlocker project

# OS Detection
OS := $(shell uname)
PYTHON := python3
POETRY := poetry
PIP := pip

ifeq ($(OS), Windows_NT)
    PYTHON := python
endif

# Targets

.PHONY: all backend proxy extension setup-backend setup-proxy setup-mitmproxy run-backend run-proxy run-extension

# Default target
all: setup-backend setup-proxy setup-mitmproxy run

# Setup targets

setup-backend:
	@echo "Setting up Backend..."
	cd backend && $(POETRY) install

setup-proxy:
	@echo "Setting up Proxy..."
	cd proxy && $(POETRY) install

setup-mitmproxy:
	@echo "Setting up mitmproxy Certificate..."
ifeq ($(OS), Windows_NT)
	@echo "For Windows: Visit http://mitm.it in your browser to download and install the certificate."
else
	@echo "For Linux/macOS: Visit http://mitm.it in your browser to download and install the certificate."
endif

# Run targets

run-backend:
	@echo "Running Backend Server..."
	cd backend && $(POETRY) run $(PYTHON) app.py

run-proxy:
	@echo "Running Proxy (mitmproxy)..."
	cd proxy && $(POETRY) run mitmdump -s modify_stream.py

run-extension:
	@echo "Make sure your extension is loaded in Chrome and enabled."

# Combined run target
run: run-backend run-proxy

# Clean targets

clean:
	@echo "Cleaning temporary files..."
	rm -rf backend/__pycache__ proxy/__pycache__

help:
	@echo "Available targets:"
	@echo "  all           - Set up and run all components"
	@echo "  setup-backend - Install backend dependencies"
	@echo "  setup-proxy   - Install proxy dependencies"
	@echo "  setup-mitmproxy - Setup mitmproxy certificate"
	@echo "  run-backend   - Run the backend server"
	@echo "  run-proxy     - Run the mitmproxy proxy"
	@echo "  run-extension - Load and enable the Chrome extension"
	@echo "  run           - Run both backend and proxy"
	@echo "  clean         - Clean temporary files"
