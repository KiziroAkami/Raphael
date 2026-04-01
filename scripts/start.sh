#!/bin/bash
# Prevent macOS idle/display sleep while the bot is running.
# Runs indefinitely until stopped — PM2 manages this process.
# No-op on non-macOS systems (caffeinate is macOS-only).
[[ "$(uname)" == "Darwin" ]] || exit 0
exec caffeinate -di
