import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest

# Make sure the project root is importable
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
