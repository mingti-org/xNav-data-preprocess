#!/usr/bin/env python3
"""Convert annotated human replays to four-view LeRobot v2.1 VLN training data."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from utils.human_replay.converter import convert


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-root",
        type=Path,
        required=True,
        help="Prepared replay root containing manifest.jsonl",
    )
    parser.add_argument(
        "--annotation-report",
        type=Path,
        required=True,
        help="Fresh navigation-process report.json for the selected run",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="New LeRobot dataset directory; existing output/staging is refused",
    )
    parser.add_argument(
        "--episode-id",
        action="append",
        help="Explicit UUID subset; by default all prepared episodes are required",
    )
    parser.add_argument("--num-workers", type=int, default=1)
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    try:
        convert(
            args.input_root,
            args.annotation_report,
            args.output_root,
            episode_ids=args.episode_id,
            num_workers=args.num_workers,
        )
    except Exception:
        logging.exception("转换失败；保留 staging 和报告，未发布目标数据集")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
