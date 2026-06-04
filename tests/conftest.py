"""pytest config and shared fixtures"""

import sys
from pathlib import Path

# Ensure project root is on PYTHONPATH
sys.path.insert(0, str(Path(__file__).parent.parent))
