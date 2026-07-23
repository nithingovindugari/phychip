"""Command-line interface for scoring, inventory, and runtime diagnostics."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .provenance import runtime_fingerprint
from .registry import list_harnesses
from .scorer import score


def _targets(value: str) -> dict:
    candidate = Path(value)
    raw = candidate.read_text(encoding="utf-8") if candidate.is_file() else value
    parsed = json.loads(raw)
    if not isinstance(parsed, dict):
        raise argparse.ArgumentTypeError("targets must decode to a JSON object")
    return parsed


def _add_score_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--circuit-id", required=True)
    parser.add_argument("--targets", required=True, type=_targets)
    parser.add_argument("--completion", required=True, type=Path)
    parser.add_argument("--eval-tol", type=float, default=0.30)
    parser.add_argument("--timeout", type=float, default=15.0)
    parser.add_argument("--require-success", action="store_true")


def _run_score(args: argparse.Namespace) -> int:
    result = score(
        args.circuit_id,
        args.targets,
        args.completion.read_text(encoding="utf-8"),
        eval_tol=args.eval_tol,
        timeout_s=args.timeout,
    )
    print(json.dumps(result.to_event(), indent=2, sort_keys=True))
    return 1 if args.require_success and not result.all_specs_met else 0


def score_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="phy-chip-score")
    _add_score_arguments(parser)
    return _run_score(parser.parse_args(argv))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="phy-chip-env")
    subparsers = parser.add_subparsers(dest="command", required=True)
    doctor = subparsers.add_parser("doctor", help="verify and fingerprint ngspice-46")
    doctor.add_argument("--ngspice-bin", default="ngspice")
    subparsers.add_parser("capabilities", help="print packaged harness metadata")
    score_parser = subparsers.add_parser("score", help="score one completion")
    _add_score_arguments(score_parser)
    args = parser.parse_args(argv)

    try:
        if args.command == "doctor":
            print(json.dumps(runtime_fingerprint(args.ngspice_bin), indent=2, sort_keys=True))
            return 0
        if args.command == "capabilities":
            print(
                json.dumps(
                    [
                        {
                            "circuit_id": info.circuit_id,
                            "tier": info.tier,
                            "fom_names": info.fom_names,
                            "spec_units": info.spec_units,
                        }
                        for info in list_harnesses()
                    ],
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0
        return _run_score(args)
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
