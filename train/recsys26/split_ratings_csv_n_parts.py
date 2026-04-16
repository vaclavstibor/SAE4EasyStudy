#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split ratings CSV into N parts without modifying source file.")
    parser.add_argument("--input", required=True, help="Path to source ratings CSV")
    parser.add_argument("--output-dir", required=True, help="Directory for output part files")
    parser.add_argument("--parts", type=int, required=True, help="Number of output parts (N)")
    parser.add_argument("--prefix", default="ratings_part", help="Output filename prefix")
    parser.add_argument("--delimiter", default=",", help="CSV delimiter (default: ',')")
    return parser.parse_args()


def count_data_rows(input_path: Path, delimiter: str) -> int:
    with input_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f, delimiter=delimiter)
        try:
            next(reader)
        except StopIteration as exc:
            raise ValueError(f"Input file is empty: {input_path}") from exc
        return sum(1 for _ in reader)


def make_part_paths(output_dir: Path, prefix: str, n_parts: int) -> list[Path]:
    pad = max(2, len(str(n_parts)))
    return [output_dir / f"{prefix}_{i:0{pad}d}.csv" for i in range(1, n_parts + 1)]


def split_csv_n_parts(input_path: Path, output_dir: Path, n_parts: int, prefix: str, delimiter: str) -> None:
    if n_parts < 2:
        raise ValueError("--parts must be >= 2")

    total_rows = count_data_rows(input_path, delimiter)
    if total_rows < n_parts:
        raise ValueError(f"Input has only {total_rows} rows, cannot split into {n_parts} parts.")

    output_dir.mkdir(parents=True, exist_ok=True)
    part_paths = make_part_paths(output_dir, prefix, n_parts)

    base_rows = total_rows // n_parts
    remainder = total_rows % n_parts
    rows_per_part = [base_rows + (1 if i < remainder else 0) for i in range(n_parts)]

    with input_path.open("r", encoding="utf-8", newline="") as in_f:
        reader = csv.reader(in_f, delimiter=delimiter)
        header = next(reader)

        writers = []
        files = []
        try:
            for path in part_paths:
                f = path.open("w", encoding="utf-8", newline="")
                files.append(f)
                writer = csv.writer(f, delimiter=delimiter)
                writer.writerow(header)
                writers.append(writer)

            current_part = 0
            rows_written_in_part = 0
            for row in reader:
                writers[current_part].writerow(row)
                rows_written_in_part += 1
                if rows_written_in_part >= rows_per_part[current_part] and current_part < n_parts - 1:
                    current_part += 1
                    rows_written_in_part = 0
        finally:
            for f in files:
                f.close()

    print(f"Input rows (without header): {total_rows}")
    print(f"Split into {n_parts} parts")
    for path, rows in zip(part_paths, rows_per_part):
        print(f"{path} rows={rows}")


if __name__ == "__main__":
    args = parse_args()
    split_csv_n_parts(
        input_path=Path(args.input),
        output_dir=Path(args.output_dir),
        n_parts=args.parts,
        prefix=args.prefix,
        delimiter=args.delimiter,
    )
