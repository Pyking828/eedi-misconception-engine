"""
Synthetic data generation (key gap vs top Kaggle solutions).

1. SynthDataGenerator: vLLM / 32B teacher MCQs for unseen misconceptions
2. MisconceptionExpander: expand misconception text for embedding separability
3. CoTDataGenerator: CoT traces for listwise SFT / GRPO
4. LLM-as-Judge: filter low-quality synth rows

Versioned prompts live under prompts/ and integrate with PromptRegistry.
"""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl

try:  # vLLM path (Blackwell sm_120 FlashInfer incompatible; see scripts/synth_data.py for transformers)
    from vllm import SamplingParams
except ImportError:  # pragma: no cover
    SamplingParams = None  # type: ignore[assignment, misc]

# ─────────────────────────────────────────────
# 1. Prompt templates (hardcoded backup; Jinja versions in prompts/)
# ─────────────────────────────────────────────

MCQ_GEN_SYSTEM = """You are an expert mathematics question designer. Generate diagnostic Multiple Choice Questions (MCQs) that reveal specific student misconceptions.

For each misconception provided, create a math question where:
- The correct answer tests the concept properly
- The incorrect answer(s) directly result from the target misconception
- Questions range from middle school to early high school level
- Use concise, precise mathematical language

Output format (JSON):
{
  "ConstructName": "<short topic name>",
  "SubjectName": "<Number|Algebra|Geometry and Measure|Data and Statistics>",
  "QuestionText": "<the math question>",
  "CorrectAnswerText": "<correct answer>",
  "WrongAnswerText": "<wrong answer stemming from the misconception>",
  "MisconceptionName": "<exact misconception text>"
}"""

MCQ_GEN_USER = """Generate {n} diagnostic MCQs for this misconception:

Misconception: {misconception}

Reference examples (similar style):
{examples}

Output one JSON object per line (JSONL format)."""

EXPAND_SYSTEM = """You are an expert mathematics educator. Expand the given misconception into a detailed explanation that helps distinguish it from similar misconceptions.

Format: "Explanation: {explanation}. Common cases: {cases}. Distinguished from: {distinctions}."
Keep it under 100 words."""

JUDGE_SYSTEM = """Evaluate whether this math question correctly diagnoses the given misconception.

Score 0-10:
- 10: Wrong answer is a direct, inevitable result of the misconception
- 7-9: Clear logical connection
- 4-6: Plausible but not airtight
- 0-3: Weak or no connection

Output format: {"score": <int>, "reason": "<one sentence>"}"""


# ─────────────────────────────────────────────
# 2. SynthDataGenerator (vLLM)
# ─────────────────────────────────────────────


class SynthDataGenerator:
    """
    Generate synthetic MCQs with local vLLM.
    Goal: cover misconceptions unseen in training for better generalization.
    """

    def __init__(
        self,
        model_name: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
        gpu_memory_utilization: float = 0.80,
        max_model_len: int = 4096,
        cache_dir: str | None = None,
    ) -> None:
        from vllm import LLM, SamplingParams

        self.llm = LLM(
            model=model_name,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            dtype="bfloat16",
            download_dir=cache_dir,
            trust_remote_code=True,
        )
        self.sampling_params = SamplingParams(
            temperature=0.8,
            top_p=0.95,
            max_tokens=512,
        )

    def generate_mcqs(
        self,
        misconceptions: list[str],
        reference_examples: dict[str, list[dict]],
        n_per_misconception: int = 5,
        output_path: str | Path | None = None,
    ) -> list[dict]:
        """Batch-generate MCQs; return list of JSONL-style records."""
        prompts = []
        misc_names = []
        for misc_name in misconceptions:
            examples = reference_examples.get(misc_name, [])
            ex_str = (
                "\n".join(json.dumps(ex, ensure_ascii=False) for ex in examples[:3])
                or "No examples available."
            )
            prompt = self._build_chat_prompt(
                MCQ_GEN_SYSTEM,
                MCQ_GEN_USER.format(
                    n=n_per_misconception,
                    misconception=misc_name,
                    examples=ex_str,
                ),
            )
            prompts.append(prompt)
            misc_names.append(misc_name)

        outputs = self.llm.generate(prompts, self.sampling_params)

        results: list[dict] = []
        for misc_name, output in zip(misc_names, outputs):
            text = output.outputs[0].text
            for line in text.strip().split("\n"):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    record.setdefault("MisconceptionName", misc_name)
                    results.append(record)
                except json.JSONDecodeError:
                    continue

        if output_path is not None:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w") as f:
                for r in results:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")

        return results

    def llm_judge(
        self,
        records: list[dict],
        threshold: float = 6.0,
    ) -> list[dict]:
        """Score synthetic rows with the same model; filter low quality."""
        prompts = []
        for rec in records:
            judge_prompt = self._build_chat_prompt(
                JUDGE_SYSTEM,
                json.dumps(rec, ensure_ascii=False),
            )
            prompts.append(judge_prompt)

        judge_params = self.llm.generate(
            prompts,
            SamplingParams(temperature=0.0, max_tokens=64),  # type: ignore[attr-defined]
        )
        filtered: list[dict] = []
        for rec, out in zip(records, judge_params):
            try:
                result = json.loads(out.outputs[0].text.strip())
                score = float(result.get("score", 0))
                if score >= threshold:
                    rec["judge_score"] = score
                    filtered.append(rec)
            except Exception:
                continue
        return filtered

    @staticmethod
    def _build_chat_prompt(system: str, user: str) -> str:
        """Qwen chat template (simplified, no tokenizer dependency)."""
        return (
            f"<|im_start|>system\n{system}<|im_end|>\n"
            f"<|im_start|>user\n{user}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )


# ─────────────────────────────────────────────
# 3. MisconceptionExpander
# ─────────────────────────────────────────────


class MisconceptionExpander:
    """Expand misconception description text (2nd-place style)."""

    def __init__(
        self,
        model_name: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
        gpu_memory_utilization: float = 0.80,
        cache_dir: str | None = None,
    ) -> None:
        from vllm import LLM, SamplingParams

        self.llm = LLM(
            model=model_name,
            gpu_memory_utilization=gpu_memory_utilization,
            dtype="bfloat16",
            download_dir=cache_dir,
            trust_remote_code=True,
        )
        self.SamplingParams = SamplingParams

    def expand(
        self,
        misconceptions: list[str],
        batch_size: int = 32,
    ) -> dict[str, str]:
        """Return {original_text: expanded_text}."""
        results: dict[str, str] = {}
        for i in range(0, len(misconceptions), batch_size):
            batch = misconceptions[i : i + batch_size]
            prompts = [
                SynthDataGenerator._build_chat_prompt(
                    EXPAND_SYSTEM,
                    f"Misconception: {m}",
                )
                for m in batch
            ]
            outputs = self.llm.generate(
                prompts,
                self.SamplingParams(temperature=0.3, max_tokens=128),
            )
            for misc, out in zip(batch, outputs):
                results[misc] = out.outputs[0].text.strip()
        return results


# ─────────────────────────────────────────────
# 4. Dataset merge utilities
# ─────────────────────────────────────────────


def merge_real_and_synth(
    real_long_df: pl.DataFrame,
    synth_records: list[dict],
    misconception_df: pl.DataFrame,
) -> pl.DataFrame:
    """Convert synth JSONL to long_df schema and concat with real data."""
    misc_id_map: dict[str, int] = {
        r["MisconceptionName"]: r["MisconceptionId"] for r in misconception_df.iter_rows(named=True)
    }

    rows = []
    for i, rec in enumerate(synth_records):
        misc_name = rec.get("MisconceptionName", "")
        misc_id = misc_id_map.get(misc_name, -1)
        if misc_id == -1:
            continue
        rows.append(
            {
                "QuestionId": -(i + 1),  # negative id marks synthetic
                "QuestionId_Answer": f"synth_{i}",
                "Answer": "A",
                "SubjectName": rec.get("SubjectName", ""),
                "ConstructName": rec.get("ConstructName", ""),
                "QuestionText": rec.get("QuestionText", ""),
                "CorrectAnswerText": rec.get("CorrectAnswerText", ""),
                "WrongAnswerText": rec.get("WrongAnswerText", ""),
                "MisconceptionId": misc_id,
                "MisconceptionName": misc_name,
                "AllText": (
                    f"Subject: {rec.get('SubjectName', '')}\n"
                    f"Topic: {rec.get('ConstructName', '')}\n"
                    f"Question: {rec.get('QuestionText', '')}\n"
                    f"Correct Answer: {rec.get('CorrectAnswerText', '')}\n"
                    f"Incorrect Answer: {rec.get('WrongAnswerText', '')}"
                ),
                "fold": -1,  # synthetic rows: train only, not in CV
                "is_synth": True,
            }
        )
    synth_df = pl.DataFrame(rows)

    if "is_synth" not in real_long_df.columns:
        real_long_df = real_long_df.with_columns(pl.lit(False).alias("is_synth"))

    return pl.concat([real_long_df, synth_df], how="diagonal")
