#!/bin/bash

# Make directory for streamlit config
mkdir -p ~/.streamlit/

# Create streamlit config with proper port binding
echo "[general]
email = \"\"
" > ~/.streamlit/credentials.toml

# Create config with explicit port from Heroku environment
echo "[server]
headless = true
port = ${PORT:-8501}
enableCORS = false
" > ~/.streamlit/config.toml
