"""pytest 配置 + 公共 fixtures"""
import sys
from pathlib import Path

# 确保项目根在 PYTHONPATH
sys.path.insert(0, str(Path(__file__).parent.parent))
