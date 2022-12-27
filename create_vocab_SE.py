## train from scratch
## enhacement: how to add more words in vocab -> find if this is possible or not

## issue: how to handle lowercase while training in both case


import argparse
from glob import glob
from pathlib import Path

from tokenizers import BertWordPieceTokenizer, ByteLevelBPETokenizer


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        default="data_dir",
        help="path to data dir containing text",
        required=True,
    )
    parser.add_argument(
        "--out_dir", default="out_dir", help="dir to store output", required=True
    )
    parser.add_argument(
        "--tokenizer_type", default="BPE", help="choose tokenizer b/w BPE and BERT"
    )
    parser.add_argument(
        "--vocab_size", default=0, type=int, help="size of vocabulary", required=True
    )
    parser.add_argument(
        "--min_fre",
        default=2,
        type=int,
        help=" min freq of a word to be considered in vocab",
    )
    parser.add_argument(
        "--model_name", default="spanELectra", help="give a name to model"
    )

    return parser


def get_tokenizer(args):
    if args.tokenizer_type == "BPE":
        ##enhancement, take lowercase,etc as User input
        tokenizer = ByteLevelBPETokenizer(lowercase=True, add_prefix_space=True)
    elif args.tokenizer_type == "BERT":
        tokenizer = BertWordPieceTokenizer()
    else:
        raise ValueError("wrong choice of tokenizer please selct among 'BPE' or 'BERT'")

    return tokenizer


def main(args):
    file_paths = [
        str(Path(x)) for x in glob(str(args.data_dir) + "*")
    ]  # comment this and uncomment below line if dir contain other type of text data than .txt ext
    # file_paths= [str(x) for x in Path().glob(str(args.data_dir)+"*")]
    print(file_paths)
    tokenizer = get_tokenizer(args)
    tokenizer.train(
        files=file_paths,
        vocab_size=args.vocab_size,
        min_frequency=args.min_fre,
        special_tokens=[
            "[CLS]",
            "[SEP]",
            "[PAD]",
            "[MASK]",
            "[UNK]",
        ],
    )

    print(tokenizer)
    # tokenizer._tokenizer.post_processor = BertProcessing(
    #     ("[sep]", tokenizer.token_to_id("[sep]")),
    #     ("[cls]", tokenizer.token_to_id("[cls]")),
    # )
    tokenizer.save(
        args.out_dir, "/" + str(args.model_name) + "_" + str(args.tokenizer_type)
    )


class tok_args:
    tokenizer_type = "BPE"
    data_dir = "/media/data_dump/Amardeep/spanElectra/data/wikitext/wikitext-2/"
    model_name = "trial"
    min_fre = 2
    vocab_size = 10000
    out_dir = "/media/data_dump/Amardeep/spanElectra/data/wikitext/tok_10k"


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)
