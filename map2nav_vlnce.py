from __future__ import annotations

import argparse

from utils.map2nav_vlnce import convert_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert R2R/RxR or ScaleVLN replay data to stable xNav data."
    )
    parser.add_argument(
        "--input-root", required=True,
        help="Replay root containing split folders, or a ScaleVLN root containing shard_* folders",
    )
    parser.add_argument("--output-root", required=True, help="Dataset output root")
    parser.add_argument(
        "--dataset-name",
        required=True,
        choices=["r2r", "rxr_guide", "scalevln"],
        help="Stable source dataset name",
    )
    parser.add_argument(
        "--split",
        required=True,
        choices=["train", "val_seen", "val_unseen"],
    )
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=None,
        help="Optional number of accepted per-instruction output episodes to write",
    )
    parser.add_argument(
        "--rxr-annotations",
        default=None,
        help=(
            "Authoritative RxR guide JSON/JSON.GZ containing instruction.language; "
            "required for rxr_guide"
        ),
    )
    parser.add_argument("--chunk-size", type=int, default=1000)
    parser.add_argument(
        "--flat-output",
        action="store_true",
        help="Write the dataset directly under output-root without appending the split name",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="Number of concurrent episode writers; output order remains manifest-deterministic",
    )
    output_mode = parser.add_mutually_exclusive_group()
    output_mode.add_argument(
        "--resume",
        action="store_true",
        help="Reuse verified per-episode outputs from an interrupted conversion",
    )
    output_mode.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace the selected output split before converting",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_root = convert_dataset(
        input_root=args.input_root,
        output_root=args.output_root,
        dataset_name=args.dataset_name,
        split=args.split,
        max_episodes=args.max_episodes,
        chunk_size=args.chunk_size,
        num_workers=args.num_workers,
        resume=args.resume,
        overwrite=args.overwrite,
        rxr_annotations=args.rxr_annotations,
        flat_output=args.flat_output,
    )
    print(f"Wrote map2nav_vlnce split to {dataset_root}")


if __name__ == "__main__":
    main()
