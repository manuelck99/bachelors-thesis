import heapq
import json
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
    open_files = [open(f.name, "r", encoding="utf-8") for f in temp_files]

    def get_line_obj(file_index):
        line = open_files[file_index].readline()
        if not line:
            return None
        obj = json.loads(line)
        return (obj["time"], file_index, line)

    heap = []
    for i in range(len(open_files)):
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

    for f in open_files:
        f.close()


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
    with open(args.input_path, "r", encoding="utf-8") as infile:
        chunk = []
        for line in infile:
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

    for f in temp_files:
        os.remove(f.name)
