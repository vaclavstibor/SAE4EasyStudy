#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge ratings CSV parts into one CSV.")
    parser.add_argument("--input-dir", required=True, help="Directory with split part files")
    parser.add_argument("--output", required=True, help="Path to merged output CSV")
    parser.add_argument("--prefix", default="ratings_part", help="Part filename prefix")
    parser.add_argument("--delimiter", default=",", help="CSV delimiter (default: ',')")
    return parser.parse_args()


def discover_part_files(input_dir: Path, prefix: str) -> list[Path]:
    parts = sorted(input_dir.glob(f"{prefix}_*.csv"))
    if not parts:
        raise ValueError(f"No part files found in {input_dir} with prefix '{prefix}'.")
    return parts


def merge_parts(input_dir: Path, output_path: Path, prefix: str, delimiter: str) -> None:
    part_files = discover_part_files(input_dir, prefix)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total_rows = 0
    expected_header = None

    with output_path.open("w", encoding="utf-8", newline="") as out_f:
        writer = csv.writer(out_f, delimiter=delimiter)

        for idx, part_path in enumerate(part_files):
            with part_path.open("r", encoding="utf-8", newline="") as in_f:
                reader = csv.reader(in_f, delimiter=delimiter)
                try:
                    header = next(reader)
                except StopIteration as exc:
                    raise ValueError(f"Empty part file: {part_path}") from exc

                if idx == 0:
                    expected_header = header
                    writer.writerow(header)
                elif header != expected_header:
                    raise ValueError(f"Header mismatch in {part_path}.")

                part_rows = 0
                for row in reader:
                    writer.writerow(row)
                    part_rows += 1
                total_rows += part_rows
                print(f"Merged {part_path} rows={part_rows}")

    print(f"Output: {output_path}")
    print(f"Total merged rows (without header): {total_rows}")


if __name__ == "__main__":
    args = parse_args()
    merge_parts(
        input_dir=Path(args.input_dir),
        output_path=Path(args.output),
        prefix=args.prefix,
        delimiter=args.delimiter,
    )
