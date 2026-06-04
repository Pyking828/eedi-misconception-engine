.PHONY: install install-dev lint format test smoke serve mcp-config clean precommit track prepare-data train-retriever synth-data train-reranker ensemble train-grpo unseen-eval

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

# ─── 提交前钩子 & 实验跟踪 ──────────────────────────────
precommit:
	pre-commit install && pre-commit run --all-files

track:
	$(PYTHON) scripts/trackio_log.py   # 记录指标历程到 Trackio，trackio show 查看

# ─── 数据准备（从 HF 镜像拉取并转换）────────────────────
prepare-data:
	HF_ENDPOINT=https://hf-mirror.com $(PYTHON) scripts/prepare_data.py

build-index:
	$(PYTHON) scripts/build_index.py --adapter-path outputs/retriever/lora_best_8b --top-k 50

# ─── 训练（按阶段）──────────────────────────────────────
train-retriever:
	$(PYTHON) scripts/retriever_baseline.py --train --fold 0

synth-data:
	$(PYTHON) scripts/synth_data.py --model r1-32b --per-mis 7 --limit-mis 0

train-reranker:
	$(PYTHON) scripts/reranker_pointwise_train.py --fold 0

ensemble:
	$(PYTHON) scripts/reranker_ensemble.py --scores outputs/reranker/scores_best31k_baselinepool_fold0_val.json outputs/reranker/scores_hn12_baselinepool_fold0_val.json --mode both

train-grpo:
	$(PYTHON) scripts/grpo_listwise.py --reward ndcg --steps 400

unseen-eval:
	$(PYTHON) scripts/unseen_eval.py

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
