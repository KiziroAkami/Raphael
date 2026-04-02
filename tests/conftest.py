"""Shared pytest configuration and fixtures."""
import sys
from pathlib import Path

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).parent.parent))


def pytest_addoption(parser):
    parser.addoption(
        "--run-integration",
        action="store_true",
        default=False,
        help="Run integration tests that call external APIs (Groq)",
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "integration: marks tests that call external APIs")
    if config.getoption("--run-integration"):
        import os
        os.environ["RUN_INTEGRATION_TESTS"] = "1"
