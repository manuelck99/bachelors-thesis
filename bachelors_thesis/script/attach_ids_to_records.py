import json
from argparse import ArgumentParser
from uuid import UUID, uuid5

NAMESPACE = UUID("c279eae10056416ba939755729b2c4f5")

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

    index = 0
    for line in in_file:
        record_id = uuid5(NAMESPACE, f"vehicle-record-{index}")
        index += 1

        record = json.loads(line)
        record["record_id"] = record_id.hex
        out_file.write(json.dumps(record))
        out_file.write("\n")

    out_file.close()
    in_file.close()
