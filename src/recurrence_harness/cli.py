from __future__ import annotations

import argparse
from pathlib import Path

from .analysis import analyze_directory
from .harness import RecurrenceHarness


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="recurrence-harness",
        description="Generate deterministic recurrence-field renders and analyze their image structure.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    render = sub.add_parser("render", help="Render one recurrence field.")
    render.add_argument("--seed", type=int, default=404)
    render.add_argument("--alpha", type=float, default=0.1745)
    render.add_argument("--size", type=int, default=240)
    render.add_argument("--steps", type=int, default=1600)
    render.add_argument("--out", default="outputs")

    sweep = sub.add_parser("sweep", help="Render a deterministic alpha sweep.")
    sweep.add_argument("--seed", type=int, default=404)
    sweep.add_argument("--alpha", type=float, nargs="+", required=True)
    sweep.add_argument("--size", type=int, default=240)
    sweep.add_argument("--steps", type=int, default=1600)
    sweep.add_argument("--out", default="outputs")

    analyze = sub.add_parser("analyze", help="Extract image-level structural measurements.")
    analyze.add_argument("input")
    analyze.add_argument("--out", default="analysis_results")
    analyze.add_argument("--pairwise", action="store_true")
    analyze.add_argument("--vector-size", type=int, default=128)

    return parser


def main() -> None:
    args = build_parser().parse_args()

    if args.command == "render":
        harness = RecurrenceHarness(alpha=args.alpha, N=args.size, steps=args.steps, out_dir=args.out)
        path = harness.single_render(args.seed)
        print(path)
        return

    if args.command == "sweep":
        harness = RecurrenceHarness(N=args.size, steps=args.steps, out_dir=args.out)
        for path in harness.alpha_sweep(args.seed, args.alpha):
            print(path)
        return

    if args.command == "analyze":
        analyze_directory(Path(args.input), Path(args.out), args.pairwise, args.vector_size)


if __name__ == "__main__":
    main()
