import json
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
        help="Path to the output records file with IDs"
    )
    args = parser.parse_args()

    in_file = open(args.input_path, mode="r", encoding="utf-8")
    out_file = open(args.output_path, mode="w", encoding="utf-8")

    record_id = 0
    for line in in_file:
        record = json.loads(line)
        record["record_id"] = record_id
        record_id += 1
        out_file.write(json.dumps(record))
        out_file.write("\n")

    in_file.close()
    out_file.close()
