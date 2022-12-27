## enchancemet: two other masking sceme


## for tokenizing, Datset and input example class
# load tokenizer, save file, load from file
# feature genration, load from file, save to file
# save in pikle file

import argparse
import json
import logging
import os
import pickle
import random
import time
from pathlib import Path

import torch
from tokenizers import BertWordPieceTokenizer, ByteLevelBPETokenizer
from tokenizers.processors import BertProcessing
from torch.utils.data import Dataset
from tqdm import tqdm
import numpy as np
from masking import get_fake_batch, get_span_masked_feature, pad_to_len
from utilis import InputExample, count_lines

logger = logging.getLogger(__name__)
import linecache


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", default="", help="output directory to store")
    parser.add_argument(
        "--feature_file", default=None, help="path of file to load features"
    )
    parser.add_argument(
        "--tokenized_file",
        default=None,
        help=" path of file to load tokenized sentemces",
    )
    parser.add_argument(
        "--save_features",
        default=False,
        type=bool,
        help="wether to save features created in file or not",
    )
    parser.add_argument(
        "--save_tokenized_text",
        default=True,
        type=bool,
        help="wether to save tokenized text or not",
    )
    parser.add_argument(
        "--tokenizer_type", default="BERT", help="type of tokenizer BPE or BERT"
    )
    parser.add_argument(
        "--tokenizer_dir",
        default=None,
        help="path to dir of tokenizer files like vocab",
    )
    parser.add_argument(
        "--raw_text_dir",
        default=None,
        help="if you want to create tokenization or feature from raw text",
    )
    parser.add_argument(
        "--mode",
        default=0,
        type=int,
        help="1 to tokenize and save in file and exit. 2 -> to tokenize, covert to features and save in file and exit. 0-> for whole function",
    )


def get_tokenizer(tokType, vocab_file, merges_file=None, lowercase=True):
    """load trained tokenizer
    tokType= BPE/BERT
    dir: path to dir containg vocab.json /vocab.txt and merges.txt (if applicable, for BPE)
    """
    tokenizer = None
    if tokType == "BPE":
        ##enhancement : deal with dash
        # vocab= os.path.join(dir,"trial BPE-vocab.json")
        # merges= os.path.join(dir,"trial BPE-merges.txt")
        ##enchansement: take lowercase and other arg as user Input
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
        # vocab= os.path.join(dir,"vocab.txt")
        tokenizer = BertWordPieceTokenizer(vocab_file, lowercase=lowercase)

    else:
        raise ValueError("wrong selection of tokenizer, select BPE or BERT")

    return tokenizer


def load_features_from_file(file_path):
    """load saved features from a pickle file"""
    infile = open(file_path, "rb")
    ex_list = pickle.load(infile)
    example = []

    for ex in ex_list:
        iex = InputExample()
        for key in ex.keys():
            if hasattr(iex, key):
                setattr(iex, key, ex[key])

        example.append(iex)

    return example


def load_tokenized_text_from_file(file_path):
    """load saved tokenized text from a pickle file"""
    infile = open(file_path, "rb")
    ex_list = pickle.load(infile)
    example = []

    for ex in ex_list:
        iex = InputExample()  ##create in term of Input example class
        for key in ex.keys():
            if hasattr(iex, key):
                setattr(iex, key, ex[key])

        example.append(iex)

    return example


def pad_to_len(list, pad, length):
    """pad a single list to len"""
    cur_len = len(list)
    if cur_len > length:
        raise ValueError("len of list is greater than padding length")

    req_len = length - cur_len

    list = list + [pad] * req_len
    assert len(list) == length
    return list


def tokenize_text(
    infile_path, tokenizer, max_seq_len, out_dir, save=True, save_name=""
):
    """tokenize raw texts into line, get ids, mask, seg_ids. save them if save= True"""
    ## issue : get tokens without padding or with any special token. try to see what happenns with token > 512 or 1024 len while using tokenizer
    pad_id = 2
    pad = "[PAD]"
    cur_len = max_seq_len
    examples = []

    if save:
        out_file = open(
            os.path.join(out_dir, save_name + "_tokenized_feature" + ".txt"),
            "w",
            encoding="utf-8",
        )
    all_token, all_ids = [], []
    logger.debug(
        "creating tokenized feature from raw text in file: {}".format(infile_path)
    )
    start_time = time.time()
    save_ex = []
    with open(infile_path, "r", encoding="utf-8") as f:
        ##enhacement : use tqdm
        for line in f:

            line = line.strip().replace("\n", " ")
            encoded = tokenizer.encode(line)
            all_token = all_token + encoded.tokens
            all_ids = all_ids + encoded.ids
            # uncomment below line and comment above if tokenizer is give output with cls and sep token
            # all_token = all_token + encoded.tokens[1:-]
            # all_ids = all_ids + encoded.ids[1:-1]

            if (
                len(all_token) < 150
            ):  ## don't make sentence with token reamining of less than 150
                continue
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

                iex = InputExample()
                iex.orig_len = cur_len
                # cls and sep and pad at 0,1,2 index with offset (0,0)
                iex.tokens = ["[CLS]"] + all_token[s:e] + ["[SEP]"]
                iex.input_id = [0] + all_ids[s:e] + [1]
                all_token = all_token[e:]  # update all token list
                all_ids = all_ids[e:]  # update all ids list
                # iex.input_mask = [1] * (cur_len + 2)  # +2 for cls and sep
                # iex.offsets = [(0,0)]+ encoded.offsets[s:e]+ [(0,0)] #no pint of offset
                # below line could be discarded
                # iex.tokens = pad_to_len(iex.tokens, "[PAD]", max_seq_len)  # padding
                # iex.input_id = pad_to_len(iex.input_id, 2, max_seq_len)  # 2 for [pad]
                # iex.input_mask = pad_to_len(
                # iex.input_mask, 0, max_seq_len
                # )  # 0 for padded
                # pad_to_len(iex.offsets,(0,0),max_seq_len) # (0,0) is dummy offset
                # iex.segment_id = []
                # iex.segment_id = pad_to_len(
                #     iex.segment_id, 1, max_seq_len
                # )  # 1 as these are single sentence

                examples.append(iex)
                if save:
                    ex = {
                        "tokens": iex.tokens,
                        "input_id": iex.input_id,
                        # "input_mask": iex.input_mask,
                        # "segment_id": iex.segment_id,
                        "orig_len": iex.orig_len,
                    }
                    out_file.write(json.dumps(ex) + "\n")

                i = e
    logger.info(
        "time taken to create toekinzed feature: {:.2f} seconds".format(
            time.time() - start_time
        )
    )
    if save:
        # pickle.dump(save_ex, out_file, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info("all tokenized feature saved in {} file".format(out_file))
        out_file.close()
    return examples


def create_masking_features(examples, masking_scheme, out_dir, mask_id=3, save=True):
    """define different masking scemes and create masking as reuired"""
    if save:
        out_file = open(os.path.join(out_dir, "masked_feature.p"), "wb")
    for iex in examples:

        if masking_scheme == "span":
            iex = get_span_masked_feature(iex, mask_id=mask_id)

        else:
            raise ValueError("wrong selection of masking scheme")
        if save:
            ex = {
                "tokens": iex.tokens,
                "input_id": iex.input_id,
                "input_mask": iex.input_mask,
                "offsets": iex.offsets,
                "segment_id": iex.segment_id,
                "orig_len": iex.orig_len,
                "lm_sentence": iex.lm_sentence,
                "target_pairs": iex.target_pairs,
                "target_spans": iex.target_spans,
            }

            pickle.dump(ex, out_file, protocol=pickle.HIGHEST_PROTOCOL)

    if save:
        out_file.close()


class MLMSpanElectraDataset(Dataset):
    def __init__(self, args):
        self.args = args
        self.examples = []
        self.tokenizer = get_tokenizer(
            args.tokenizer_type, args.vocab_file, args.merges_file, args.lowercase
        )
        self.max_seq_len = args.max_seq_len
        self.max_span_len = args.max_span_len

        if args.mask_feature_file:
            logger.info("loading feature from masked feature file")
            self.examples = load_features_from_file(args.mask_feature_file)

        elif args.tokenized_file:
            logger.info("loading tokenized text from  tokenized file file")
            self.examples = load_tokenized_text_from_file(args.tokenized_file)
            print("len of examples", len(self.examples))

        else:
            logger.info("tokenization and feature creation from raw text data")
            self.examples = tokenize_text(
                args.raw_text_dir,
                self.tokenizer,
                max_seq_len=args.max_seq_len,
                out_dir=args.out_dir,
                save=args.save_tokenized_text,
                save_name=args.save_name,
            )

        if args.occu is not None:
            logger.info("using only a subset of dataset")
            self.examples = self.examples[: args.occu]

        logger.info("total number of examples: {}".format(self.__len__()))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        """define tensor as required and return them as per scheme or model
        return example[idx]
        """
        if self.args.mask_feature_file:
            curr_ex = self.examples[idx]
        else:
            curr_ex = get_span_masked_feature(
                example=self.examples[idx],
                mask_id=self.args.mask_id,
                pad_token=self.args.pad_token,
                mask_ratio=self.args.mask_ratio,
                pad_token_id=self.args.pad_token_id,
                max_seq_len=self.max_seq_len,
                max_span_len=self.max_span_len,
            )
        ex = {
            "input_id": curr_ex.input_id,
            "input_mask": curr_ex.input_mask,
            "segment_id": curr_ex.segment_id,
            "orig_len": curr_ex.orig_len,
            "lm_sentence": curr_ex.lm_sentence,
            "pairs": curr_ex.target_pairs,
            "spans": curr_ex.target_spans,
        }
        assert len(ex["input_id"]) == self.max_seq_len
        return ex

    def collate_fun(self, batch):
        """to modify and make constant number of pairs at batch level for Dataloders"""
        dummy_id = 0
        pairs = [x["pairs"] for x in batch]
        spans = [x["spans"] for x in batch]
        ids = [x["input_id"] for x in batch]
        imask = [x["input_mask"] for x in batch]
        sid = [x["segment_id"] for x in batch]
        ori_len = [x["orig_len"] for x in batch]
        lm_sent = [x["lm_sentence"] for x in batch]

        mx_pairs = max(len(x) for x in pairs)
        for i, (s, p) in enumerate(zip(spans, pairs)):
            pairs[i] = pad_to_len(
                p, [dummy_id, dummy_id], mx_pairs
            )  # dummy_id is to generate fake pairs to satisfy equal lenght
            spans[i] = pad_to_len(
                s, [self.args.pad_token_id] * self.max_span_len, mx_pairs
            )

        new_batch = {
            "input_id": torch.tensor(ids, dtype=torch.long),
            "input_mask": torch.tensor(imask, dtype=torch.long),
            "segment_id": torch.tensor(sid, dtype=torch.long),
            "orig_len": torch.tensor(ori_len, dtype=torch.long),
            "lm_sentence": torch.tensor(lm_sent, dtype=torch.long),
            "pairs": torch.tensor(pairs, dtype=torch.long),
            "spans": torch.tensor(spans, dtype=torch.long),
        }

        return new_batch


class SpanELectraDataset(Dataset):
    def __init__(self, args):
        self.args = args
        self.examples = []
        self.tokenizer = get_tokenizer(
            args.tokenizer_type, args.vocab_file, args.merges_file, args.lowercase
        )
        self.max_seq_len = args.max_seq_len

        self.generator = args.generator_path
        self.max_span_len = args.max_span_len

        if args.mask_feature_file:
            logger.info("loading feature from masked feature file")
            self.examples = load_features_from_file(args.mask_feature_file)

        elif args.tokenized_file:
            logger.info("loading tokenized text from  tokenized file file")
            self.examples = load_tokenized_text_from_file(args.tokenized_file)
            print("len of examples", len(self.examples))

        else:
            logger.info("tokenization and feature creation from raw text data")
            self.examples = tokenize_text(
                args.raw_text_dir,
                self.tokenizer,
                max_seq_len=args.max_seq_len,
                out_dir=args.out_dir,
                save=args.save_tokenized_text,
                save_name=args.save_name,
            )

        if args.occu is not None:
            logger.info("using only a subset of dataset")
            self.examples = self.examples[: args.occu]

        logger.info("total number of examples: {}".format(self.__len__()))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        if self.args.mask_feature_file:
            curr_ex = self.examples[idx]
        else:
            curr_ex = get_span_masked_feature(
                example=self.examples[idx],
                mask_id=self.args.mask_id,
                pad_token=self.args.pad_token,
                mask_ratio=self.args.mask_ratio,
                pad_token_id=self.args.pad_token_id,
                max_seq_len=self.max_seq_len,
                max_span_len=self.max_span_len,
            )

        ex = {
            "input_id": curr_ex.input_id,
            "input_mask": curr_ex.input_mask,
            "segment_id": curr_ex.segment_id,
            "orig_len": curr_ex.orig_len,
            "lm_sentence": curr_ex.lm_sentence,
            "pairs": curr_ex.target_pairs,
            "spans": curr_ex.target_spans,
        }
        assert len(ex["input_id"]) == self.max_seq_len
        return ex

    def collate_fun(self, batch):
        nw_batch = get_fake_batch(
            batch=batch,
            generator=self.generator,
            max_seq_len=self.max_seq_len,
            vocab_size=self.tokenizer.get_vocab_size(),
            pad_token_id=self.args.pad_token_id,
            max_span_len=self.max_span_len,
        )

        return nw_batch


class SpanELectraLazyDataset(Dataset):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.file = args.inFile
        st = time.time()
        logger.info("loading data with argument\n{}".format(args.__dict__))
        logger.info("counting total datapoint ")
        self.len = count_lines(self.file)
        if self.args.occur is not None:
            self.len = self.args.occur
        logger.info("{} datapoint. counting took {}".format(self.len, time.time() - st))

    def __getitem__(self, idx):
        """define tensor as required and return them as per scheme or model
        return example[idx]
        """
        cur_ex = json.loads(linecache.getline(self.file, idx + 1))
        ex = InputExample()
        # ex.input_id = cur_ex["input_id"]
        # ex.orig_len = cur_ex["orig_len"]
        ex.input_id = cur_ex
        ex.orig_len = len(cur_ex) - 2

        # if "tokens" in cur_ex.keys():
        #     ex.tokens = cur_ex["tokens"]
        curr_ex = get_span_masked_feature(
            example=ex,
            mask_id=self.args.mask_id,
            pad_token=self.args.pad_token,
            mask_ratio=self.args.mask_ratio,
            pad_token_id=self.args.pad_token_id,
            max_seq_len=self.args.max_seq_len,
            max_span_len=self.args.max_span_len,
        )
        ex = {
            "input_id": curr_ex.input_id,
            "input_mask": curr_ex.input_mask,
            "segment_id": curr_ex.segment_id,
            "orig_len": curr_ex.orig_len,
            "lm_sentence": curr_ex.lm_sentence,
            "pairs": curr_ex.target_pairs,
            "spans": curr_ex.target_spans,
        }
        assert len(ex["input_id"]) == self.args.max_seq_len
        return ex

    def __len__(self):
        return self.len

    def collate_fun(self, batch):
        """to modify and make constant number of pairs at batch level for Dataloders"""
        dummy_id = 0
        pairs = [x["pairs"] for x in batch]
        spans = [x["spans"] for x in batch]
        ids = [x["input_id"] for x in batch]
        imask = [x["input_mask"] for x in batch]
        sid = [x["segment_id"] for x in batch]
        ori_len = [x["orig_len"] for x in batch]
        lm_sent = [x["lm_sentence"] for x in batch]

        mx_pairs = max(len(x) for x in pairs)
        for i, (s, p) in enumerate(zip(spans, pairs)):
            pairs[i] = pad_to_len(
                p, [dummy_id, dummy_id], mx_pairs
            )  # dummy_id is to generate fake pairs to satisfy equal lenght
            spans[i] = pad_to_len(
                s, [self.args.pad_token_id] * self.args.max_span_len, mx_pairs
            )

        new_batch = {
            "input_id": torch.tensor(ids, dtype=torch.long),
            "input_mask": torch.tensor(imask, dtype=torch.long),
            "segment_id": torch.tensor(sid, dtype=torch.long),
            "orig_len": torch.tensor(ori_len, dtype=torch.long),
            "lm_sentence": torch.tensor(lm_sent, dtype=torch.long),
            "pairs": torch.tensor(pairs, dtype=torch.long),
            "spans": torch.tensor(spans, dtype=torch.long),
        }

        return new_batch


class BinaryIndexedDataset(Dataset):
    def __init__(self, args):
        super().__init__()
        self.args = args
        input_file_path = args.inFile
        self.data_file = open(input_file_path, "rb", buffering=0)
        if os.path.exists(input_file_path[:-4] + "_indexing.json"):
            with open(input_file_path[:-4] + "_indexing.json", "r") as f:
                idx_data = f.read()
                idx_data = json.loads(idx_data)
                self.offsets = idx_data["offsets"]
                self.array_sizes = idx_data["sizes"]
                self.len = idx_data["len"]
        else:
            raise FileNotFoundError("indexed file not found")
        if self.args.occur is not None:
            self.len = self.args.occur
        self.dtype = np.int32
        self.byte_size = 4

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        # print(idx)
        array_size = self.array_sizes[idx]
        a = np.empty(array_size, dtype=np.int32)
        self.data_file.seek(self.offsets[idx] * self.byte_size)
        self.data_file.readinto(a)
        # print(a)
        input_id = a.tolist()
        orig_len = len(input_id) - 2
        # print(input_id)
        ex = InputExample()
        ex.input_id = input_id
        ex.orig_len = orig_len
        assert ex.input_id[0] == 0 and ex.input_id[-1] == 1
        curr_ex = get_span_masked_feature(
            example=ex,
            mask_id=self.args.mask_id,
            pad_token=self.args.pad_token,
            mask_ratio=self.args.mask_ratio,
            pad_token_id=self.args.pad_token_id,
            max_seq_len=self.args.max_seq_len,
            max_span_len=self.args.max_span_len,
        )
        ex = {
            "input_id": curr_ex.input_id,
            "input_mask": curr_ex.input_mask,
            "segment_id": curr_ex.segment_id,
            "orig_len": curr_ex.orig_len,
            "lm_sentence": curr_ex.lm_sentence,
            "pairs": curr_ex.target_pairs,
            # "spans": curr_ex.target_spans,
            "labels": curr_ex.labels,
        }
        assert len(ex["input_id"]) == self.args.max_seq_len
        return ex

    def collate_fun(self, batch):
        """to modify and make constant number of pairs at batch level for Dataloders"""
        dummy_id = 0
        pairs = [x["pairs"] for x in batch]
        # spans = [x["spans"] for x in batch]
        ids = [x["input_id"] for x in batch]
        imask = [x["input_mask"] for x in batch]
        sid = [x["segment_id"] for x in batch]
        ori_len = [x["orig_len"] for x in batch]
        lm_sent = [x["lm_sentence"] for x in batch]
        labels = [x["labels"] for x in batch]
        mx_pairs = max(len(x) for x in pairs)
        # for i, (s, p) in enumerate(zip(spans, pairs)):
        #     pairs[i] = pad_to_len(
        #         p, [dummy_id, dummy_id], mx_pairs
        #     )  # dummy_id is to generate fake pairs to satisfy equal lenght
        #     spans[i] = pad_to_len(
        #         s, [self.args.pad_token_id] * self.args.max_span_len, mx_pairs
        #     )
        for i, p in enumerate(pairs):
            pairs[i] = pad_to_len(
                p, [dummy_id, dummy_id], mx_pairs
            )  # dummy_id is to generate fake pairs to satisfy equal lenght

        new_batch = {
            "input_id": torch.tensor(ids, dtype=torch.long),
            "input_mask": torch.tensor(imask, dtype=torch.long),
            "segment_id": torch.tensor(sid, dtype=torch.long),
            "orig_len": torch.tensor(ori_len, dtype=torch.long),
            "lm_sentence": torch.tensor(lm_sent, dtype=torch.long),
            "pairs": torch.tensor(pairs, dtype=torch.long),
            # "spans": torch.tensor(spans, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

        return new_batch


class CachedBinaryIndexedDataset(BinaryIndexedDataset):
    def __init__(self, args, chunk_size):
        super().__init__(args)
        self.args = args
        input_file_path = args.inFile
        self.data_file = open(input_file_path, "rb", buffering=0)
        if os.path.exists(input_file_path[:-4] + "_indexing.json"):
            with open(input_file_path[:-4] + "_indexing.json", "r") as f:
                idx_data = f.read()
                idx_data = json.loads(idx_data)
                self.offsets = idx_data["offsets"]
                self.array_sizes = idx_data["sizes"]
                self.len = idx_data["len"]
        else:
            raise FileNotFoundError("indexed file not found")
        if self.args.occur is not None:
            self.len = self.args.occur
        self.cache = {}
        self.chunk_size = chunk_size
        self.byte_size = 4

    def __len__(self):
        return self.len

    def get_chunk_id(self, idx):
        return int(idx // self.chunk_size)

    def get_chunk(self, start_idx, end_idx):
        self.data_file.seek(int(self.offsets[start_idx] * self.byte_size))
        curr_list = []
        for idx in range(start_idx, end_idx):
            array_size = self.array_sizes[idx]
            a = np.empty(array_size, dtype=np.int32)
            self.data_file.readinto(a)
            curr_list.append(a.tolist())
        return curr_list

    def __getitem__(self, idx):
        # print(idx)
        chunk_id = self.get_chunk_id(idx)
        cid = idx % self.chunk_size
        if chunk_id not in self.cache.keys():
            self.cache[chunk_id] = {}
            sid = idx
            eid = min(idx + self.chunk_size, self.len)
            self.cache[chunk_id]["data"] = self.get_chunk(sid, eid)
            self.cache[chunk_id]["count"] = 0

        ex = InputExample()
        ex.input_id = self.cache[chunk_id]["data"][cid]
        self.cache[chunk_id]["count"] += 1
        if self.cache[chunk_id]["count"] == self.chunk_size:
            del self.cache[chunk_id]
        ex.orig_len = len(ex.input_id) - 2
        # print("ex last",ex.input_id[0], ex.input_id[-1])
        assert ex.input_id[0] == 0 and ex.input_id[-1] == 1
        curr_ex = get_span_masked_feature(
            example=ex,
            mask_id=self.args.mask_id,
            pad_token=self.args.pad_token,
            mask_ratio=self.args.mask_ratio,
            pad_token_id=self.args.pad_token_id,
            max_seq_len=self.args.max_seq_len,
            max_span_len=self.args.max_span_len,
        )
        ex = {
            "input_id": curr_ex.input_id,
            "input_mask": curr_ex.input_mask,
            "segment_id": curr_ex.segment_id,
            "orig_len": curr_ex.orig_len,
            "lm_sentence": curr_ex.lm_sentence,
            "pairs": curr_ex.target_pairs,
            # "spans": curr_ex.target_spans,
            "labels": curr_ex.labels,
        }
        assert len(ex["input_id"]) == self.args.max_seq_len
        return ex

    # def collate_fun(self, batch):
    #     """ to modify and make constant number of pairs at batch level for Dataloders
    #     """
    #     dummy_id = 0
    #     pairs = [x["pairs"] for x in batch]
    #     spans = [x["spans"] for x in batch]
    #     ids = [x["input_id"] for x in batch]
    #     imask = [x["input_mask"] for x in batch]
    #     sid = [x["segment_id"] for x in batch]
    #     ori_len = [x["orig_len"] for x in batch]
    #     lm_sent = [x["lm_sentence"] for x in batch]

    #     mx_pairs = max(len(x) for x in pairs)
    #     for i, (s, p) in enumerate(zip(spans, pairs)):
    #         pairs[i] = pad_to_len(
    #             p, [dummy_id, dummy_id], mx_pairs
    #         )  # dummy_id is to generate fake pairs to satisfy equal lenght
    #         spans[i] = pad_to_len(
    #             s, [self.args.pad_token_id] * self.args.max_span_len, mx_pairs
    #         )

    #     new_batch = {
    #         "input_id": torch.tensor(ids, dtype=torch.long),
    #         "input_mask": torch.tensor(imask, dtype=torch.long),
    #         "segment_id": torch.tensor(sid, dtype=torch.long),
    #         "orig_len": torch.tensor(ori_len, dtype=torch.long),
    #         "lm_sentence": torch.tensor(lm_sent, dtype=torch.long),
    #         "pairs": torch.tensor(pairs, dtype=torch.long),
    #         "spans": torch.tensor(spans, dtype=torch.long),
    #     }

    #     return new_batch


class DiscBIDataset(Dataset):
    def __init__(self, args, chunk_size):
        super().__init__()
        self.args = args
        input_file_path = args.inFile
        self.data_file = open(input_file_path, "rb", buffering=0)
        if os.path.exists(input_file_path[:-4] + "_indexing.json"):
            with open(input_file_path[:-4] + "_indexing.json", "r") as f:
                idx_data = f.read()
                idx_data = json.loads(idx_data)
                self.offsets = idx_data["offsets"]
                self.array_sizes = idx_data["sizes"]
                self.len = idx_data["len"] // 3
        else:
            raise FileNotFoundError("indexed file not found")
        if self.args.occur is not None:
            self.len = self.args.occur
        self.cache = {}
        self.chunk_size = chunk_size
        self.byte_size = 4
        self.pad_token_id = self.args.pad_token_id

    def __len__(self):
        return self.len

    def get_chunk_id(self, idx):
        return int(idx // self.chunk_size)

    def get_chunk(self, start_idx, end_idx):
        self.data_file.seek(int(self.offsets[start_idx] * self.byte_size))
        curr_list = []
        for idx in range(start_idx, end_idx, 3):
            cl = []
            for i in range(3):
                array_size = self.array_sizes[idx + i]
                a = np.empty(array_size, dtype=np.int32)
                self.data_file.readinto(a)
                cl.append(a.tolist())

            pair = cl[-1]
            assert len(pair) % 2 == 0
            nwp = [[pair[2 * i], pair[2 * i + 1]] for i in range(len(pair) // 2)]
            cl[-1] = nwp
            assert len(cl) == 3
            curr_list.append(cl)
        return curr_list

    def __getitem__(self, idx):
        # print(idx)
        chunk_id = self.get_chunk_id(idx)
        cid = idx % self.chunk_size
        if chunk_id not in self.cache.keys():
            self.cache[chunk_id] = {}
            sid = 3 * idx
            eid = 3 * min(idx + self.chunk_size, self.len)
            self.cache[chunk_id]["data"] = self.get_chunk(sid, eid)
            self.cache[chunk_id]["count"] = 0
            # if len(self.cache[chunk_id]["data"]) != self.chunk_size:
            #     print(len(self.cache[chunk_id]["data"]), "chunk_size data mismatch")

        ex = InputExample()
        # if len(self.cache[chunk_id]["data"]) !=self.chunk_size:
        #     print(len(self.cache[chunk_id]["data"]), "chun len is not bs chunk id, cid, len ", chunk_id, cid, self.len)
        # if cid>= len(self.cache[chunk_id]["data"]):
        #     print(len(self.cache[chunk_id]["data"]), "cid is greater than data len")

        ex.input_id = self.cache[chunk_id]["data"][cid][0]
        ex.orig_len = len(ex.input_id) - 2
        ex.labels = self.cache[chunk_id]["data"][cid][1]
        ex.target_pairs = self.cache[chunk_id]["data"][cid][2]
        self.cache[chunk_id]["count"] += 1
        if self.cache[chunk_id]["count"] == len(self.cache[chunk_id]["data"]):
            del self.cache[chunk_id]

        ex.segment_id = pad_to_len([], 1, self.args.max_seq_len)
        ex.input_mask = [1] * len(ex.input_id)
        ex.input_mask = pad_to_len(ex.input_mask, 0, self.args.max_seq_len)
        ex.input_id = pad_to_len(ex.input_id, self.pad_token_id, self.args.max_seq_len)
        ex.labels = pad_to_len(ex.labels, self.args.ignore_label, self.args.max_seq_len)

        iex = {
            "input_id": ex.input_id,
            "input_mask": ex.input_mask,
            "segment_id": ex.segment_id,
            "orig_len": ex.orig_len,
            # "lm_sentence": ex.lm_sentence,
            "pairs": ex.target_pairs,
            # "spans": curr_ex.target_spans,
            "labels": ex.labels,
        }
        assert len(iex["input_id"]) == self.args.max_seq_len
        return iex

    def collate_fun(self, batch):
        """to modify and make constant number of pairs at batch level for Dataloders"""
        dummy_id = 0
        pairs = [x["pairs"] for x in batch]
        # spans = [x["spans"] for x in batch]
        ids = [x["input_id"] for x in batch]
        imask = [x["input_mask"] for x in batch]
        sid = [x["segment_id"] for x in batch]
        ori_len = [x["orig_len"] for x in batch]
        # lm_sent = [x["lm_sentence"] for x in batch]
        labels = [x["labels"] for x in batch]
        mx_pairs = max(len(x) for x in pairs)

        for i, p in enumerate(pairs):
            pairs[i] = pad_to_len(
                p, [dummy_id, dummy_id], mx_pairs
            )  # dummy_id is to generate fake pairs to satisfy equal lenght

        new_batch = {
            "input_id": torch.tensor(ids, dtype=torch.long),
            "input_mask": torch.tensor(imask, dtype=torch.long),
            "segment_id": torch.tensor(sid, dtype=torch.long),
            "orig_len": torch.tensor(ori_len, dtype=torch.long),
            "pairs": torch.tensor(pairs, dtype=torch.long),
            # "spans": torch.tensor(spans, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

        return new_batch
