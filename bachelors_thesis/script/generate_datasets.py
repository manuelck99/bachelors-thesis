import json
import os
from argparse import ArgumentParser

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "-i", "--input-path",
        type=str,
        required=True,
        help="Path to the input records file"
    )
    parser.add_argument(
        "-o", "--output-path",
        type=str,
        required=True,
        help="Path to the output datasets folder"
    )
    parser.add_argument(
        "-s", "--sizes",
        type=int,
        required=True,
        nargs="+",
        help="Sizes of the datasets to generate"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Generate all dataset"
    )
    parser.add_argument(
        "--annotated",
        action="store_true",
        help="Generate annotated dataset"
    )
    args = parser.parse_args()

    dataset_sizes = sorted(args.sizes)
    dataset_files = {size: open(os.path.join(args.output_path, f"records-{size}.json"), mode="w", encoding="utf-8") for
                     size in dataset_sizes}
    dataset_line_counts = {size: 0 for size in dataset_sizes}

    if args.annotated:
        annotated_path = os.path.join(args.output_path, "records-annotated.json")
        annotated_file = open(annotated_path, mode="w", encoding="utf-8")

    if args.all:
        all_path = os.path.join(args.output_path, "records-all.json")
        all_file = open(all_path, mode="w", encoding="utf-8")

    with open(args.input_path, mode="r", encoding="utf-8") as in_file:
        for line in in_file:
            record = json.loads(line)
            is_annotated = record["vehicle_id"] is not None

            if args.all:
                all_file.write(line)
            if args.annotated and is_annotated:
                annotated_file.write(line)

            for dataset_size in dataset_sizes:
                if is_annotated or dataset_line_counts[dataset_size] < dataset_size:
                    dataset_files[dataset_size].write(line)
                    if not is_annotated:
                        dataset_line_counts[dataset_size] += 1

    for file in dataset_files.values():
        file.close()

    if args.annotated:
        annotated_file.close()

    if args.all:
        all_file.close()
