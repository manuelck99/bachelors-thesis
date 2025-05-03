import heapq
import json
import math
import os
import sys
import tempfile
from argparse import ArgumentParser


def write_sorted_chunk(chunk):
    chunk.sort(key=lambda line: json.loads(line)["time"])
    temp_file = tempfile.NamedTemporaryFile(delete=False, mode="w+t")
    for line in chunk:
        temp_file.write(line)
    temp_file.flush()
    return temp_file


def merge_sorted_chunks(temp_files, output_path):
    files = [open(temp_file.name, "r", encoding="utf-8") for temp_file in temp_files]

    def get_line_obj(file_index):
        line = files[file_index].readline()
        if not line:
            return None
        obj = json.loads(line)
        return (obj["time"], file_index, line)

    heap = []
    for i in range(len(files)):
        entry = get_line_obj(i)
        if entry:
            heapq.heappush(heap, entry)

    with open(output_path, "w", encoding="utf-8") as outfile:
        while heap:
            _, file_index, line = heapq.heappop(heap)
            outfile.write(line)
            entry = get_line_obj(file_index)
            if entry:
                heapq.heappush(heap, entry)

    for file in files:
        file.close()


def are_records_sorted(path):
    prev_time = -math.inf
    with open(path, mode="r", encoding="utf-8") as file:
        for line in file:
            record = json.loads(line)
            if prev_time > record["time"]:
                return False
            else:
                prev_time = record["time"]
    return True


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
        help="Path to the sorted output records file"
    )
    parser.add_argument(
        "-c", "--chunk-size",
        type=int,
        required=False,
        default=100_000,
        help="Chunk size to use"
    )
    args = parser.parse_args()

    temp_files = []
    with open(args.input_path, "r", encoding="utf-8") as in_file:
        chunk = []
        for line in in_file:
            chunk.append(line)
            if len(chunk) >= args.chunk_size:
                temp_file = write_sorted_chunk(chunk)
                temp_files.append(temp_file)
                chunk = []

        # Handle remaining lines
        if chunk:
            temp_file = write_sorted_chunk(chunk)
            temp_files.append(temp_file)

    merge_sorted_chunks(temp_files, args.output_path)

    for temp_file in temp_files:
        os.remove(temp_file.name)

    if not are_records_sorted(args.output_path):
        print("Sorting of records unsuccessful", file=sys.stderr)
