from create_feature_SE import get_rob_tokenizer, get_rob_features, yield_chunks
from glob import glob
from pathlib import Path
from utilis import count_lines
from multiprocessing import Pool
from tqdm import tqdm
import random, json
import argparse
from dataset import TextDatasetWriter, BinaryIndexDatasetWriter


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tokenizer_type", default="BPE", help="type of tokenizer BPE or BERT"
    )
    parser.add_argument("--mode", default="bin", help="store in txt/bin file")
    parser.add_argument(
        "--vocab_file",
        default=None,
        help="path to vocab.json file created after tokenization process",
    )
    parser.add_argument(
        "--merges_file",
        default=None,
        help="path to merges.txt file created after tokenization process",
    )
    parser.add_argument(
        "--out_dir",
        default=None,
        help="path to output dir to store created features",
    )
    parser.add_argument(
        "--in_dir",
        default=None,
        help="file path to input_dir containing all file",
    )
    parser.add_argument(
        "--max_seq_len", default=512, type=int, help="max seq len of features"
    )
    parser.add_argument(
        "--chunk_size",
        default=1000000,
        type=int,
        help="size of chunk to be processed by one worker",
    )
    parser.add_argument("--workers", default=30, type=int, help="number of workers")
    parser.add_argument(
        "--valid_split", default=0.2, type=float, help="percentage of validation data"
    )

    return parser


def main(
    in_dir,
    out_dir,
    tokenizer_type,
    max_seq_len,
    vocab_file,
    merges_file=None,
    lowercase=True,
    valid_split=0.2,
    chunk_size=100000,
    workers=20,
    mode="bin",
    tokenizer=None,
):
    if out_dir is None:
        out_dir = in_dir
    file_paths = [str(Path(x)) for x in glob(str(in_dir) + "*")]
    if tokenizer is None:
        tokenizer = get_rob_tokenizer(
            vocab_file=vocab_file, merges_file=merges_file, lowercase=False
        )
    output = [[]] * workers

    def on_return(features):
        # print("callback")
        worker_id, examples = features
        output[worker_id] = examples

    if mode == "bin":
        valid_output = BinaryIndexDatasetWriter(dir_path=out_dir, file_name="valid")
        train_output = BinaryIndexDatasetWriter(dir_path=out_dir, file_name="train")
    else:
        valid_output = TextDatasetWriter(dir_path=out_dir, file_name="valid")
        train_output = TextDatasetWriter(dir_path=out_dir, file_name="train")
    for in_file_path in file_paths:
        output = [[]] * workers
        tt = count_lines(in_file_path)
        for lines in tqdm(
            yield_chunks(in_file_path, chunk_size), total=tt // chunk_size
        ):
            pool = Pool()
            size = (
                (len(lines) // workers)
                if len(lines) % workers == 0
                else (1 + (len(lines) // workers))
            )
            for i in range(workers):
                start = i * size
                pool.apply_async(
                    get_rob_features,
                    args=(
                        i,
                        lines[start : start + size],
                        tokenizer,
                        max_seq_len,
                    ),
                    callback=on_return,
                )
            pool.close()
            pool.join()
            total_feat = sum(len(x) for x in output)
            valid_split_count = int(valid_split * total_feat)
            valid_idx = random.sample(list(range(total_feat)), k=valid_split_count)
            idx = 0
            for examples in output:
                for ex in examples:
                    if idx in valid_idx:
                        valid_output.write_line(ex["input_id"])
                    else:
                        train_output.write_line(ex["input_id"])
                    idx += 1

    train_output.close_writer()
    valid_output.close_writer()


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(
        in_dir=args.in_dir,
        out_dir=args.out_dir,
        tokenizer_type=args.tokenizer_type,
        max_seq_len=args.max_seq_len,
        vocab_file=args.vocab_file,
        merges_file=args.merges_file,
        chunk_size=args.chunk_size,
        workers=args.workers,
        valid_split=args.valid_split,
        mode=args.mode,
    )

# python create_feature_from_multiple_files.py --tokenizer_type "BPE" --vocab_file "/media/data_dump/Amardeep/spanElectra/data/wikitext/tok_10k/trial BPE-vocab.json" --merges_file "/media/data_dump/Amardeep/spanElectra/data/wikitext/tok_10k/trial BPE-merges.txt" --out_dir "/media/data_dump/Amardeep/test_fol/" --in_dir "/media/data_dump/Amardeep/spanElectra/data/wikitext/wikitext-2/" --max_seq_len 512 --workers 10 --chunk_size 10000 --valid_split 0.3
