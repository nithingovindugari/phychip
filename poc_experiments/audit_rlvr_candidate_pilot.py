#!/usr/bin/env python3
"""Reproduce admission and decontamination checks for a promoted RLVR pilot."""

from __future__ import annotations

import argparse
import csv
import glob
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "phy-chip-env" / "src"))

from phy_chip_env.provenance import runtime_fingerprint  # noqa: E402
from phy_chip_env.scorer import SCORER_SEMANTICS, score  # noqa: E402


def norm(value: str) -> str:
    return re.sub(r"\s+", " ", (value or "").casefold()).strip()


def ngrams(value: str, size: int = 8) -> set[str]:
    words = norm(value).split()
    return {" ".join(words[index:index + size]) for index in range(max(0, len(words) - size + 1))}


def row_texts(row: dict) -> tuple[list[str], list[str]]:
    prompts: list[str] = []
    references: list[str] = []
    for key in ("spec_prompt", "prompt", "instruction", "question", "spec"):
        if row.get(key):
            prompts.append(str(row[key]))
    for key in ("reference_netlist", "netlist"):
        if row.get(key):
            references.append(str(row[key]))
    if isinstance(row.get("messages"), list):
        for message in row["messages"]:
            if message.get("role") == "user":
                prompts.append(str(message.get("content", "")))
            elif message.get("role") == "assistant":
                references.append(str(message.get("content", "")))
    return prompts, references


def load_jsonl_corpus(paths: list[Path], excluded_root: Path) -> tuple[list[tuple[str, str]], set[str], int]:
    prompts: list[tuple[str, str]] = []
    references: set[str] = set()
    rows = 0
    for path in sorted(set(paths)):
        try:
            path.resolve().relative_to(excluded_root.resolve())
            continue
        except ValueError:
            pass
        try:
            lines = path.read_text(errors="ignore").splitlines()
        except OSError:
            continue
        for line in lines:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(row, dict):
                continue
            rows += 1
            row_prompts, row_references = row_texts(row)
            prompts.extend((str(path.relative_to(ROOT)), text) for text in row_prompts if text)
            references.update(norm(text) for text in row_references if text)
    return prompts, references, rows


def is_quarantined_candidate_path(path: Path) -> bool:
    """Candidate-authoring pools are diagnostics, never part of training overlap."""
    return any(part.startswith("rlvr_candidates_") for part in path.parts)


def max_overlap(candidate: str, corpus: list[tuple[str, str]]) -> tuple[float, str | None]:
    candidate_grams = ngrams(candidate)
    best = (0.0, None)
    for source, text in corpus:
        score_value = len(candidate_grams & ngrams(text)) / max(1, len(candidate_grams))
        if score_value > best[0]:
            best = (score_value, source)
    return best


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--overlap-block", type=float, default=0.40)
    parser.add_argument("--expected-family-count", type=int, default=None)
    parser.add_argument("--expected-per-family", type=int, default=1)
    args = parser.parse_args()

    candidate_path = args.candidates.resolve()
    candidates = [json.loads(line) for line in candidate_path.read_text().splitlines() if line.strip()]
    if not candidates:
        raise SystemExit("candidate file is empty")

    training_patterns = [
        "data/sft_*/**/*.jsonl", "data/sft_*/*.jsonl",
        "data/rlvr_*/**/*.jsonl", "data/rlvr_*/*.jsonl",
        "data/*grpo*.jsonl", "data/*_grpo/*.jsonl",
        "data/final_data_clean/*.jsonl", "data/sft_multispec*/*.jsonl",
        "final_artifacts/datasets/**/*.jsonl",
    ]
    training_paths: list[Path] = []
    for pattern in training_patterns:
        training_paths.extend(Path(path) for path in glob.glob(str(ROOT / pattern), recursive=True))
    training_paths = [path for path in training_paths if not is_quarantined_candidate_path(path)]
    training_prompts, training_refs, training_rows = load_jsonl_corpus(
        training_paths, candidate_path.parent
    )

    eval_paths = [
        ROOT / "eval_sets" / "bench_v1" / "bench_v1.jsonl",
        ROOT / "eval_sets" / "phychip_bench_v2" / "bench_v2.jsonl",
    ]
    eval_prompts, eval_refs, eval_rows = load_jsonl_corpus(eval_paths, Path("/nonexistent"))
    analog_path = Path.home() / ".cache" / "phychip" / "analogcoder_problem_set.tsv"
    if analog_path.exists():
        with analog_path.open(errors="ignore") as handle:
            for index, row in enumerate(csv.reader(handle, delimiter="\t")):
                eval_prompts.append((f"analogcoder:{index}", " ".join(row)))

    fingerprint = runtime_fingerprint()
    ids: set[str] = set()
    topologies: set[str] = set()
    family_counts: Counter[str] = Counter()
    rows_report: list[dict] = []
    failures: list[str] = []
    for candidate in candidates:
        spec_id = candidate["spec_id"]
        if spec_id in ids:
            failures.append(f"duplicate spec_id: {spec_id}")
        ids.add(spec_id)
        topologies.add(candidate["topology_fingerprint"])
        family_counts[candidate["circuit_id"]] += 1
        result = score(
            candidate["circuit_id"],
            candidate["target_measurements"],
            f"```spice\n{candidate['reference_netlist']}\n```",
            timeout_s=25,
        )
        train_overlap, train_source = max_overlap(candidate["spec_prompt"], training_prompts)
        eval_overlap, eval_source = max_overlap(candidate["spec_prompt"], eval_prompts)
        exact_training_ref = norm(candidate["reference_netlist"]) in training_refs
        exact_eval_ref = norm(candidate["reference_netlist"]) in eval_refs
        if candidate.get("scorer_semantics") != SCORER_SEMANTICS:
            failures.append(f"{spec_id}: stale scorer semantics")
        if candidate.get("verifier_hash") != fingerprint["verifier_hash"]:
            failures.append(f"{spec_id}: stale verifier hash")
        if not result.all_specs_met or result.anticheat_blocked:
            failures.append(f"{spec_id}: reference no longer passes scorer")
        if train_overlap >= args.overlap_block or eval_overlap >= args.overlap_block:
            failures.append(f"{spec_id}: 8-gram overlap over threshold")
        if exact_training_ref or exact_eval_ref:
            failures.append(f"{spec_id}: exact reference match")
        rows_report.append({
            "spec_id": spec_id,
            "circuit_id": candidate["circuit_id"],
            "all_specs_met": result.all_specs_met,
            "anticheat_blocked": result.anticheat_blocked,
            "max_train_prompt_8gram_overlap": round(train_overlap, 6),
            "max_train_overlap_source": train_source,
            "max_eval_prompt_8gram_overlap": round(eval_overlap, 6),
            "max_eval_overlap_source": eval_source,
            "exact_training_reference_match": exact_training_ref,
            "exact_eval_reference_match": exact_eval_ref,
        })

    if len(topologies) != len(candidates):
        failures.append("pilot topology fingerprints are not unique")
    if args.expected_family_count is not None and len(family_counts) != args.expected_family_count:
        failures.append(
            f"pilot contains {len(family_counts)} families; expected {args.expected_family_count}"
        )
    unexpected_counts = {
        family: count
        for family, count in sorted(family_counts.items())
        if count != args.expected_per_family
    }
    if unexpected_counts:
        failures.append(
            "pilot family counts do not match expected-per-family="
            f"{args.expected_per_family}: {unexpected_counts}"
        )

    report = {
        "schema": "phychip-rlvr-candidate-audit-v1",
        "passed": not failures,
        "candidate_count": len(candidates),
        "family_count": len(family_counts),
        "family_counts": dict(sorted(family_counts.items())),
        "topology_fingerprint_count": len(topologies),
        "scorer_semantics": SCORER_SEMANTICS,
        "verifier_hash": fingerprint["verifier_hash"],
        "training_rows_scanned": training_rows,
        "evaluation_rows_scanned": eval_rows + (24 if analog_path.exists() else 0),
        "overlap_block": args.overlap_block,
        "failures": failures,
        "rows": rows_report,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "rows"}, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
