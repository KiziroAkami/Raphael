#!/bin/bash
# Wrap the bot with caffeinate so macOS does not idle-sleep while it runs.
# -d  prevent display sleep
# -i  prevent idle sleep (the most important flag)
exec caffeinate -di "$(dirname "$0")/../.venv/bin/python" "$(dirname "$0")/../main.py"
