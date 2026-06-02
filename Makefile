.PHONY: install install-dev lint format test smoke serve mcp-config clean

PYTHON := python
PIP := pip
UV := uv

# ─── 安装 ────────────────────────────────────────────
install:
	$(PIP) install -q --cache-dir /root/autodl-tmp/pip_cache -e .

install-dev:
	$(PIP) install -q --cache-dir /root/autodl-tmp/pip_cache -e ".[dev]"

# ─── 代码质量 ─────────────────────────────────────────
lint:
	ruff check src/ eval/ service/ mcp_server/ scripts/ tests/

format:
	ruff format src/ eval/ service/ mcp_server/ scripts/ tests/
	black src/ eval/ service/ mcp_server/ scripts/ tests/

# ─── 测试 ────────────────────────────────────────────
test:
	pytest tests/ -v --tb=short

smoke:
	@echo "=== 冒烟测试：不依赖 Kaggle 数据 ==="
	pytest tests/test_evaluator.py tests/test_router.py tests/test_listwise.py -v

# ─── 数据准备（需要先下载 Kaggle 数据）──────────────────
eda:
	$(PYTHON) scripts/00_eda.py

build-index:
	$(PYTHON) scripts/05_build_index.py

# ─── 训练（按阶段）──────────────────────────────────────
train-retriever:
	$(PYTHON) scripts/01_retriever_baseline.py --train --fold 0

synth-data:
	$(PYTHON) scripts/02_synth_data.py --n 5

train-reranker:
	$(PYTHON) scripts/03_reranker_train.py --stage both --fold 0

train-grpo:
	$(PYTHON) scripts/04_grpo_train.py --reward ndcg_gain

# ─── 服务 ────────────────────────────────────────────
serve:
	uvicorn service.app:app --host 0.0.0.0 --port 6006 --reload

# ─── MCP 配置输出 ────────────────────────────────────
mcp-config:
	@echo '{'
	@echo '  "mcpServers": {'
	@echo '    "eedi-misconception-engine": {'
	@echo '      "command": "python",'
	@echo '      "args": ["$(shell pwd)/mcp_server/server.py"],'
	@echo '      "env": {'
	@echo '        "HF_HOME": "/root/autodl-tmp/hf_cache",'
	@echo '        "EEDI_CONFIG": "$(shell pwd)/configs/base.yaml"'
	@echo '      }'
	@echo '    }'
	@echo '  }'
	@echo '}'

# ─── 清理 ────────────────────────────────────────────
clean:
	find . -name "*.pyc" -delete
	find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache .ruff_cache
	@echo "清理完成"
