import argparse
import json
from multiprocessing import Pool
import random
import sys

from tokenizers import BertWordPieceTokenizer, ByteLevelBPETokenizer
from tokenizers.processors import BertProcessing
from tqdm import tqdm
from transformers import RobertaTokenizerFast, BertTokenizer, RobertaTokenizer
from utilis import InputExample, count_lines
from dataset import TextDatasetWriter, BinaryIndexDatasetWriter


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tokenizer_type", default="BPE", help="type of tokenizer BPE or BERT"
    )
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
        "--out_file",
        default=None,
        help="path to output file to store created features",
    )
    parser.add_argument(
        "--in_file",
        default=None,
        help="file path to train or valid or test file",
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
    return parser


def get_tokenizer(tokType, vocab_file, merges_file=None, lowercase=True):
    """load trained tokenizer
    tokType= BPE/BERT
    dir: path to dir containg vocab.json /vocab.txt and merges.txt (if applicable, for BPE)
    """
    tokenizer = None
    if tokType == "BPE":
        tokenizer = ByteLevelBPETokenizer(
            vocab_file, merges_file, lowercase=lowercase, add_prefix_space=True
        )
        ## we cahnged token wrt Roberta <s> or </s> replaced by [cls] and [sep]
        ## uncomment comment below line if you want encoded sentence to be padded with [cls] and [sep]
        # tokenizer._tokenizer.post_processor = BertProcessing(
        #     ("[SEP]", tokenizer.token_to_id("[SEP]")),
        #     ("[CLS]", tokenizer.token_to_id("[CLS]")),
        # )

    elif tokType == "BERT":
        tokenizer = BertWordPieceTokenizer(vocab_file, lowercase=lowercase)

    else:
        raise ValueError("wrong selection of tokenizer, select BPE or BERT")

    return tokenizer


def yield_chunks(file_path, chunk_size):
    inFile = open(file_path, "r")
    curr_chunk = []
    for line in inFile:
        curr_chunk.append(line.strip())
        if len(curr_chunk) == chunk_size:
            # print("yield", len(curr_chunk))
            yield curr_chunk
            curr_chunk = []
    yield curr_chunk


def get_features(worker_id, lines, tokenizer, max_seq_len):
    cls_token = "[CLS]"
    cls_id = 0
    sep_token = "[SEP]"
    sep_id = 1
    examples = []
    all_token, all_ids = [], []
    for line in lines:
        if line == "\n" or line == " " or line == "":
            continue
        line = line.strip().replace("\n", " ")
        encoded = tokenizer.encode(line)
        all_token = all_token + encoded.tokens
        all_ids = all_ids + encoded.ids
        # uncomment below line and comment above if tokenizer give output with cls and sep token
        # all_token = all_token + encoded.tokens[1:-]
        # all_ids = all_ids + encoded.ids[1:-1]
        # if (
        #     len(all_token) <
        # ):  ## don't make sentence with token reamining of less than 150
        #     continue
        i = 0
        while i < len(all_token):
            if random.random() < 0.05:  # curr len is decided by some randomness
                cur_len = random.randint(1, max_seq_len - 2)
            else:
                cur_len = max_seq_len - 2  # -2 for cls and sep token

            s = i
            e = i + cur_len
            if i + cur_len > len(
                all_token
            ):  # change curr len if not sufficint amount of token is left
                e = len(all_token)
                cur_len = e - s

            iex = {}
            iex["orig_len"] = cur_len
            # cls and sep and pad at 0,1,2 index with offset (0,0)
            iex["tokens"] = [cls_token] + all_token[s:e] + [sep_token]
            iex["input_id"] = [cls_id] + all_ids[s:e] + [sep_id]
            all_token = all_token[e:]  # update all token list
            all_ids = all_ids[e:]  # update all ids list
            examples.append(iex)
            i = e
    return worker_id, examples


def get_rob_tokenizer(vocab_file, merges_file=None, lowercase=True):
    """load rob trained tokenizer
    dir: path to dir containg vocab.json /vocab.txt and merges.txt (if applicable, for BPE)
    """
    tokenizer = RobertaTokenizer(
        vocab_file=vocab_file,
        merges_file=merges_file,
        sep_token="[SEP]",
        cls_token="[CLS]",
        pad_token="[PAD]",
        mask_token="[MASK]",
        add_prefix_space=True,
        do_lower_case=True,
    )
    return tokenizer


def get_rob_features(worker_id, lines, tokenizer, max_seq_len):
    cls_token = "[CLS]"
    cls_id = 0
    sep_token = "[SEP]"
    sep_id = 1
    examples = []
    all_ids = []
    for line in lines:
        if line == "\n" or line == " " or line == "":
            continue  # rob tokenizer will give error on emnpty line or space
        line = line.strip().replace("\n", " ").lower()
        curr_ids = tokenizer.encode(line)
        all_ids = all_ids + curr_ids[1:-1]
        i = 0
        while i < len(all_ids):
            if random.random() < 0.05:  # curr len is decided by some randomness
                cur_len = random.randint(1, max_seq_len - 2)
            else:
                cur_len = max_seq_len - 2  # -2 for cls and sep token
            s = i
            e = i + cur_len
            if i + cur_len > len(
                all_ids
            ):  # change curr len if not sufficint amount of token is left
                e = len(all_ids)
                cur_len = e - s
            iex = {}
            # iex["orig_len"] = cur_len
            # cls and sep and pad at 0,1,2 index with offset (0,0)
            iex["input_id"] = [cls_id] + all_ids[s:e] + [sep_id]
            # iex["tokens"] = tokenizer.convert_ids_to_tokens(iex["input_id"])
            # all_token = all_token[e:]  # update all token list
            all_ids = all_ids[e:]  # update all ids list
            examples.append(iex)
            i = e
    # print(worker_id, examples)
    return worker_id, examples


def main(
    in_file_path,
    out_file_path,
    tokenizer_type,
    max_seq_len,
    vocab_file,
    merges_file=None,
    lowercase=True,
    chunk_size=100000,
    workers=20,
    mode="bin",
):
    output = [[]] * workers
    outFile = open(out_file_path, "w")
    tokenizer = get_rob_tokenizer(
        vocab_file=vocab_file, merges_file=merges_file, lowercase=False
    )

    def on_return(features):
        # print("callback")
        worker_id, examples = features
        output[worker_id] = examples

    tt = count_lines(in_file_path)
    for lines in tqdm(yield_chunks(in_file_path, chunk_size), total=tt // chunk_size):
        # print("yoeld line len", len(lines))
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
        for examples in output:
            for ex in examples:
                outFile.write(json.dumps(ex) + "\n")
    outFile.close()


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(
        in_file_path=args.in_file,
        out_file_path=args.out_file,
        tokenizer_type=args.tokenizer_type,
        max_seq_len=args.max_seq_len,
        vocab_file=args.vocab_file,
        merges_file=args.merges_file,
        chunk_size=args.chunk_size,
        workers=args.workers,
    )

# python create_feature.py --tokenizer_type "BPE" --vocab_file "/media/data_dump/Amardeep/spanElectra/data/wikitext/tok_10k/trial BPE-vocab.json" --merges_file "/media/data_dump/Amardeep/spanElectra/data/wikitext/tok_10k/trial BPE-merges.txt" --out_file "/media/data_dump/Amardeep/spanElectra/out/jfeat/train.txt" --in_file "/media/data_dump/Amardeep/spanElectra/data/wikitext/wikitext-2/wiki.train.tokens" --max_seq_len 512 --workers 10 --chunk_size 10000
