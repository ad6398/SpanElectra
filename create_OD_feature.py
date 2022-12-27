from glob import glob
from pathlib import Path
from utilis import count_lines
from multiprocessing import Pool
from tqdm import tqdm
import random, json
import argparse
from dataset import TextDatasetWriter, BinaryIndexDatasetWriter
from transformers import BertTokenizer
from create_feature_SE import yield_chunks
import math
import numpy as np
from masking import merge_intervals, trim_spans


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


def corrupt_input_labels(ids, pairs, vocab_size):
    sent = ids[:]
    labels = [0] * len(sent)
    idx_list = []
    for s, e in pairs:
        idx_list += range(s, e + 1)
    pos_list = random.sample(idx_list, len(idx_list) // 2)
    for idx in pos_list:
        cv = random.randint(0, vocab_size - 1)
        if cv != sent[idx]:
            sent[idx] = cv
            labels[idx] = 1

    return [sent, labels]


def get_labels_corrupt_feat(
    example,
    max_seq_len,
    vocab_size,
    geometric_dist=0.2,
    max_span_len=15,
    mask_ratio=0.2,
    span_lower=1,
    span_upper=10,
):
    lens = list(range(span_lower, span_upper + 1))
    len_distrib = (
        [
            geometric_dist * (1 - geometric_dist) ** (i - span_lower)
            for i in range(span_lower, span_upper + 1)
        ]
        if geometric_dist >= 0
        else None
    )
    tmpSum = sum(len_distrib)
    len_distrib = [x / tmpSum for x in len_distrib]
    sent_length = len(example) - 2
    mask_num = math.ceil(sent_length * mask_ratio)
    mask = set()
    target_pairs = []

    while len(mask) < mask_num:
        span_len = np.random.choice(lens, p=len_distrib)
        span_len = min(span_len, max_span_len)
        start_tok = np.random.choice(sent_length)
        start_pos = 1 + start_tok
        ## enchancement: here subpices of words are also considered, this could be enchanced to conside whole word. IMO this method will be usful wrt BPE tokenzers and for word piece that method will be good

        if start_pos in mask:
            continue

        itr = start_pos
        current_span_len = 0
        end_pos = itr
        while (
            itr <= sent_length and current_span_len < span_len and len(mask) < mask_num
        ):
            end_pos = itr
            current_span_len += 1
            mask.add(itr)
            itr += 1

        target_pairs.append([start_pos, end_pos])
    target_pairs = merge_intervals(target_pairs)  # merge colliding interval
    assert len(mask) == sum([e - s + 1 for s, e in target_pairs])
    target_pairs = trim_spans(target_pairs, max_span_len)
    return corrupt_input_labels(example, target_pairs, vocab_size) + [target_pairs]


def get_features(worker_id, lines, tokenizer, max_seq_len):
    cls_token = "[CLS]"
    sep_token = "[SEP]"
    examples = []
    all_ids = []
    for line in lines:
        if line == "\n" or line == " " or line == "":
            continue  # rob tokenizer will give error on emnpty line or space
        line = line.strip().replace("\n", " ").lower()
        curr_ids = tokenizer.tokenize(line)
        all_ids = all_ids + curr_ids
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
            # iex["orig_len"] = cur_len
            # cls and sep and pad at 0,1,2 index with offset (0,0)
            iex = [cls_token] + all_ids[s:e] + [sep_token]
            iex = tokenizer.convert_tokens_to_ids(iex)
            all_ids = all_ids[e:]  # update all ids list
            examples.append(iex)
            i = e

    for i, ex in enumerate(examples):
        # print(ex,"dfhsakjfs")
        examples[i] = get_labels_corrupt_feat(
            ex,
            max_seq_len=max_seq_len,
            vocab_size=tokenizer.vocab_size,
            max_span_len=15,
        )
        # print(ex)
    # print(worker_id, examples)
    return worker_id, examples


def main(
    in_dir,
    out_dir,
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
        tokenizer = BertTokenizer.from_pretrained(vocab_file, do_lower_case=True)
    output = [[]] * workers

    def on_return(features):
        worker_id, examples = features
        output[worker_id] = examples

    if mode == "bin":
        valid_output = BinaryIndexDatasetWriter(dir_path=out_dir, file_name="valid")
        train_output = BinaryIndexDatasetWriter(dir_path=out_dir, file_name="train")
    else:
        valid_output = TextDatasetWriter(dir_path=out_dir, file_name="valid")
        train_output = TextDatasetWriter(dir_path=out_dir, file_name="train")

    get_features(0, ["I am amardeep"], tokenizer, 10)
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
                    get_features,
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
                        valid_output.write_line(ex[0])
                        valid_output.write_line(ex[1])
                        pair = [x for y in ex[2] for x in y]
                        valid_output.write_line(pair)
                    else:
                        train_output.write_line(ex[0])
                        train_output.write_line(ex[1])
                        pair = [x for y in ex[2] for x in y]
                        train_output.write_line(pair)
                    idx += 1
            print("lines written ", idx)

    train_output.close_writer()
    valid_output.close_writer()


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(
        in_dir=args.in_dir,
        out_dir=args.out_dir,
        max_seq_len=args.max_seq_len,
        vocab_file=args.vocab_file,
        chunk_size=args.chunk_size,
        workers=args.workers,
        valid_split=args.valid_split,
        mode=args.mode,
    )
    # python create_OD_feature.py --vocab_file "/media/data_dump/Amardeep/bert-base-uncased-vocab.txt"  --out_dir "/media/data_dump/Amardeep/test_fol/od_bert" --in_dir "/media/data_dump/Amardeep/spanElectra/data/wikitext/wikitext-2/" --max_seq_len 512 --workers 10 --chunk_size 100000 --valid_split 0.3
