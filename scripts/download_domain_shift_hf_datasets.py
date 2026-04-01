#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from datasets import load_dataset
from huggingface_hub import hf_hub_download, snapshot_download


PromptBuilder = Callable[[dict[str, Any]], str | None]


@dataclass(frozen=True)
class DatasetSpec:
    domain: str
    hf_path: str
    config: str | None
    split: str
    prompt_builder: PromptBuilder
    note: str
    file_format: str | None = None
    data_files: tuple[str, ...] = ()


def _clean(text: str | None) -> str | None:
    if text is None:
        return None
    text = " ".join(str(text).split())
    return text if text else None


def _build_gsm8k(example: dict[str, Any]) -> str | None:
    return _clean(example.get("question"))


def _build_squad(example: dict[str, Any]) -> str | None:
    context = _clean(example.get("context"))
    question = _clean(example.get("question"))
    if not context or not question:
        return None
    return f"Context: {context}\nQuestion: {question}"


def _build_dialogsum(example: dict[str, Any]) -> str | None:
    dialogue = _clean(example.get("dialogue"))
    if not dialogue:
        return None
    return f"Summarize the following conversation:\n{dialogue}"


def _build_opus(example: dict[str, Any]) -> str | None:
    translation = example.get("translation") or {}
    src = _clean(translation.get("en"))
    tgt_lang = "fr"
    if not src:
        return None
    return f"Translate the following English text to {tgt_lang}:\n{src}"


def _build_finance_qa(example: dict[str, Any]) -> str | None:
    question = _clean(example.get("question"))
    context = _clean(example.get("context"))
    company = _clean(example.get("company")) or _clean(example.get("ticker"))
    if not question:
        return None
    if context and company:
        return f"Company: {company}\nContext: {context}\nFinance question: {question}"
    if context:
        return f"Context: {context}\nFinance question: {question}"
    if company:
        return f"Company: {company}\nFinance question: {question}"
    return f"Finance question: {question}"


def _build_medical(example: dict[str, Any]) -> str | None:
    question = _clean(example.get("Question"))
    if question:
        return question
    input_text = _clean(example.get("input"))
    if input_text:
        return input_text
    return _clean(example.get("prompt"))


def _build_legal_qa(example: dict[str, Any]) -> str | None:
    question = _clean(example.get("question"))
    if not question:
        return None
    return f"Legal question: {question}"


def _build_leetcode(example: dict[str, Any]) -> str | None:
    title = _clean(example.get("title"))
    description = _clean(example.get("content")) or _clean(example.get("description"))
    if title and description:
        return f"{title}\n{description}"
    return description or title


DATASETS: list[DatasetSpec] = [
    DatasetSpec(
        domain="math",
        hf_path="openai/gsm8k",
        config="main",
        split="train",
        prompt_builder=_build_gsm8k,
        note="grade-school math reasoning",
    ),
    DatasetSpec(
        domain="general_qa",
        hf_path="rajpurkar/squad_v2",
        config=None,
        split="train",
        prompt_builder=_build_squad,
        note="general-domain question answering",
    ),
    DatasetSpec(
        domain="summarization",
        hf_path="knkarthick/dialogsum",
        config=None,
        split="train",
        prompt_builder=_build_dialogsum,
        note="dialog summarization",
        file_format="csv",
        data_files=("train.csv",),
    ),
    DatasetSpec(
        domain="translation",
        hf_path="Helsinki-NLP/opus_books",
        config=None,
        split="train",
        prompt_builder=_build_opus,
        note="English-to-French translation",
        file_format="parquet",
        data_files=("en-fr/train-00000-of-00001.parquet",),
    ),
    DatasetSpec(
        domain="finance",
        hf_path="SheikhIrtiza/finance_QA_dataset",
        config=None,
        split="train",
        prompt_builder=_build_finance_qa,
        note="finance-domain question answering",
        file_format="csv",
        data_files=("Financial-QA-10k.csv",),
    ),
    DatasetSpec(
        domain="medical",
        hf_path="FreedomIntelligence/medical-o1-reasoning-SFT",
        config=None,
        split="train",
        prompt_builder=_build_medical,
        note="medical reasoning",
        file_format="json",
        data_files=("medical_o1_sft.json",),
    ),
    DatasetSpec(
        domain="legal",
        hf_path="isaacus/legal-rag-qa",
        config=None,
        split="train",
        prompt_builder=_build_legal_qa,
        note="legal question answering",
        file_format="json",
        data_files=("qa.jsonl",),
    ),
    DatasetSpec(
        domain="code",
        hf_path="greengerong/leetcode",
        config=None,
        split="train",
        prompt_builder=_build_leetcode,
        note="coding/problem-solving",
        file_format="json",
        data_files=("leetcode-train.jsonl",),
    ),
]


def _load_source_dataset(spec: DatasetSpec, limit: int):
    split = f"{spec.split}[:{limit}]"
    if spec.file_format is None:
        return load_dataset(spec.hf_path, spec.config, split=split)

    if len(spec.data_files) == 1:
        local_file = hf_hub_download(
            repo_id=spec.hf_path,
            repo_type="dataset",
            filename=spec.data_files[0],
        )
        data_files: dict[str, str | list[str]] = {spec.split: local_file}
    else:
        local_dir = snapshot_download(
            repo_id=spec.hf_path,
            repo_type="dataset",
            allow_patterns=list(spec.data_files),
        )
        resolved_files = [str(Path(local_dir) / relpath) for relpath in spec.data_files]
        data_files = {spec.split: resolved_files}

    return load_dataset(spec.file_format, data_files=data_files, split=split)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _load_domain_rows(
    spec: DatasetSpec,
    per_domain: int,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    ds = _load_source_dataset(spec, per_domain * 3)

    rows: list[dict[str, Any]] = []
    for idx, example in enumerate(ds):
        prompt = spec.prompt_builder(example)
        prompt = _clean(prompt)
        if not prompt:
            continue
        rows.append(
            {
                "domain": spec.domain,
                "prompt": prompt,
                "source_dataset": spec.hf_path,
                "source_config": spec.config,
                "source_split": spec.split,
                "source_index": idx,
            }
        )
        if len(rows) >= per_domain:
            break

    unique_rows = len(rows)
    if unique_rows == 0:
        raise RuntimeError(
            f"{spec.domain}: collected 0 usable rows from {spec.hf_path}"
        )

    rng = random.Random(seed)
    resampled_rows = 0
    if unique_rows < per_domain:
        print(
            f"[warn] {spec.domain}: only collected {unique_rows} unique rows from "
            f"{spec.hf_path}; resampling to reach {per_domain}"
        )
        base_rows = list(rows)
        while len(rows) < per_domain:
            template = dict(rng.choice(base_rows))
            template["resampled"] = True
            template["resample_id"] = resampled_rows
            rows.append(template)
            resampled_rows += 1

    for row in rows[:unique_rows]:
        row["resampled"] = False

    rng.shuffle(rows)
    return rows, {
        "unique_rows": unique_rows,
        "resampled_rows": resampled_rows,
    }


def _build_stable_rows(rows_by_domain: dict[str, list[dict[str, Any]]], length: int) -> list[dict[str, Any]]:
    stable_rows: list[dict[str, Any]] = []
    for domain, rows in rows_by_domain.items():
        for i in range(min(length, len(rows))):
            row = dict(rows[i])
            row["scenario"] = f"stable_{domain}"
            row["sequence_id"] = i
            stable_rows.append(row)
    return stable_rows


def _build_block_rows(
    rows_by_domain: dict[str, list[dict[str, Any]]],
    domains: list[str],
    block_size: int,
    rounds: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    cursor = {domain: 0 for domain in domains}
    seq = 0
    for _ in range(rounds):
        for domain in domains:
            for _ in range(block_size):
                row = dict(rows_by_domain[domain][cursor[domain]])
                row["scenario"] = f"block_{block_size}"
                row["sequence_id"] = seq
                rows.append(row)
                cursor[domain] += 1
                seq += 1
    return rows


def _build_alternating_rows(
    rows_by_domain: dict[str, list[dict[str, Any]]],
    domains: list[str],
    pair_span: int,
    rounds: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    cursor = {domain: 0 for domain in domains}
    seq = 0
    for _ in range(rounds):
        for domain in domains:
            for _ in range(pair_span):
                row = dict(rows_by_domain[domain][cursor[domain]])
                row["scenario"] = f"alternating_{pair_span}"
                row["sequence_id"] = seq
                rows.append(row)
                cursor[domain] += 1
                seq += 1
    return rows


def _build_random_rows(
    rows_by_domain: dict[str, list[dict[str, Any]]],
    domains: list[str],
    length: int,
    seed: int,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    cursor = {domain: 0 for domain in domains}
    rows: list[dict[str, Any]] = []
    for seq in range(length):
        domain = rng.choice(domains)
        row = dict(rows_by_domain[domain][cursor[domain]])
        row["scenario"] = "random_uniform"
        row["sequence_id"] = seq
        rows.append(row)
        cursor[domain] += 1
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download and organize Hugging Face datasets for domain-shift serving experiments."
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/domain_shift_hf"),
    )
    parser.add_argument(
        "--per-domain",
        type=int,
        default=512,
        help="Number of normalized prompts to keep per domain.",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=64,
        help="Block size for block-shift scenarios.",
    )
    parser.add_argument(
        "--stable-length",
        type=int,
        default=128,
        help="Rows per stable scenario file.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
    )
    args = parser.parse_args()

    out_dir = args.out_dir
    raw_dir = out_dir / "by_domain"
    scenario_dir = out_dir / "scenarios"
    raw_dir.mkdir(parents=True, exist_ok=True)
    scenario_dir.mkdir(parents=True, exist_ok=True)

    rows_by_domain: dict[str, list[dict[str, Any]]] = {}
    manifest: dict[str, Any] = {"domains": {}, "scenarios": {}}

    for spec in DATASETS:
        rows, stats = _load_domain_rows(spec, args.per_domain, args.seed)
        rows_by_domain[spec.domain] = rows
        _write_jsonl(raw_dir / f"{spec.domain}.jsonl", rows)
        manifest["domains"][spec.domain] = {
            "hf_path": spec.hf_path,
            "config": spec.config,
            "split": spec.split,
            "file_format": spec.file_format,
            "data_files": list(spec.data_files),
            "rows": len(rows),
            "unique_rows": stats["unique_rows"],
            "resampled_rows": stats["resampled_rows"],
            "note": spec.note,
        }

    domains = list(rows_by_domain.keys())
    stable_rows = _build_stable_rows(rows_by_domain, args.stable_length)
    block_rounds = max(1, args.per_domain // max(1, args.block_size * len(domains)))
    block_rows = _build_block_rows(rows_by_domain, domains, args.block_size, block_rounds)
    alternating_rows = _build_alternating_rows(rows_by_domain, domains, 2, args.per_domain // (2 * len(domains)))
    random_rows = _build_random_rows(rows_by_domain, domains, min(args.per_domain, 256), args.seed)

    scenario_files = {
        "stable": stable_rows,
        f"block_{args.block_size}": block_rows,
        "alternating_2": alternating_rows,
        "random_uniform": random_rows,
    }
    for name, rows in scenario_files.items():
        _write_jsonl(scenario_dir / f"{name}.jsonl", rows)
        manifest["scenarios"][name] = {"rows": len(rows)}

    with (out_dir / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"Wrote dataset bundle to: {out_dir}")


if __name__ == "__main__":
    main()
