## utiities code for like reading file, and stuffs
import os
import pickle
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt
from sklearn.metrics import f1_score
from configuration_span_electra import SpanElectraConfig
import argparse


def save_stats(save_dir, name, **kwargs):
    out_file = open(os.path.join(save_dir, name + ".p"), "wb")
    save_lst = {}
    for key, val in kwargs.items():
        save_lst[key] = val

    pickle.dump(save_lst, out_file, protocol=pickle.HIGHEST_PROTOCOL)
    out_file.close()


def plot2(
    ts, loss=None, acc=None, x1=None, y1=None, x2=None, y2=None, figName="fig", dire=""
):
    fig = plt.figure(figsize=(15, 5), dpi=300)
    ts = int(ts)
    if loss is not None:
        if acc is not None:
            fig1 = fig.add_subplot(1, 2, 1)
        else:
            fig1 = fig.add_subplot(1, 1, 1)
        # plt.xticks([])
        for cp in loss.keys():
            fig1.plot(list(loss[cp]), label=str(cp))
        if x1 is not None:
            plt.xlabel(x1)
        if y1 is not None:
            plt.ylabel(y1)
        plt.legend()

    if acc is not None:
        if loss is not None:
            fig2 = fig.add_subplot(1, 2, 2)
        else:
            fig2 = fig.add_subplot(1, 1, 1)
        # plt.xticks([])
        for cp in acc.keys():
            fig2.plot(list(acc[cp]), label=str(cp))
        if x2 is not None:
            plt.xlabel(x2)
        if y2 is not None:
            plt.ylabel(y2)
        plt.legend()

    fig.tight_layout()
    plt.savefig(dire + str(figName) + ".png")
    plt.savefig(dire + str(figName) + ".svg")


def get_mlm_loss_out_sbo_labels(logits, labels=None, ignore_labels=2, pad_token_id=2):
    mlm_loss = None
    lab_size = labels.size()
    clf_labels = None
    pred_tokens = None
    logits = logits.view(-1, logits.size(-1))
    # assert logits.size(-1)== self.gen_config.vocab_size
    # pred_tokens = logits.argmax(dim=1)
    if labels is not None:
        mlm_loss, pred_tokens = get_ceLoss_pre(
            logits=logits, labels=labels, ignore_idx=ignore_labels, reduce=True
        )
        labels = labels.view(-1)

        pred_tokens = pred_tokens.view(-1)
        clf_labels = torch.zeros(
            labels.size(0), dtype=torch.long, device=logits.device
        )  # 0 for orig toekns
        wrong_pre = pred_tokens != labels
        masked_pos = labels == pad_token_id
        clf_labels[wrong_pre] = 1  # 1 for replaced token
        clf_labels[masked_pos] = ignore_labels  # dummy posiition with pad_token_id=2
    pred_tokens = pred_tokens.view(lab_size)
    clf_labels = clf_labels.view(lab_size)
    return mlm_loss, pred_tokens, clf_labels


def get_disc_input_at_labels(
    input_ids, gen_tokens, span_labels, pairs, dummy_id, pad_token_id=2, ignore_label=2
):
    device = input_ids.device
    clf_inputs = input_ids.clone()
    all_token_labels = torch.zeros(clf_inputs.size(), dtype=torch.long, device=device)
    pad_mask = clf_inputs == pad_token_id  # get pos of padded tokens
    all_token_labels[
        pad_mask
    ] = ignore_label  # put ignore label for padded token for classi
    for i in range(clf_inputs.size(0)):
        for j in range(pairs[i].size(0)):
            s, e = pairs[i][j][0], pairs[i][j][1]
            if s == e and s == dummy_id:  # got dummy input
                break
            # print("span before",clf_inputs[i][s:e+1] )
            # print("s,e gen_toklen", s, e, gen_tokens[i][j].size())
            clf_inputs[i][s : e + 1] = gen_tokens[i][j][0 : e - s + 1]
            all_token_labels[i][s : e + 1] = span_labels[i][j][0 : e - s + 1]
            # print("span after",clf_inputs[i][s:e+1] )
            # print("span_labels", all_token_labels[i][s:e+1])
            # print("orig span", spans[i][j])

    assert clf_inputs.size() == input_ids.size()
    assert all_token_labels.size() == input_ids.size()
    return clf_inputs, all_token_labels


def get_disc_in_disc_labels(
    input_ids, pred_tokens, orig_labels, pad_token_id, ignore_label
):
    device = input_ids.device
    clf_inputs = input_ids.clone().detach()
    mask = orig_labels != pad_token_id
    clf_inputs[mask] = pred_tokens[mask]
    nt_eq = clf_inputs != orig_labels
    at_labels = torch.zeros(clf_inputs.size(), dtype=torch.long, device=device)
    at_labels[mask & nt_eq] = 1
    pad = input_ids == pad_token_id
    at_labels[pad] = ignore_label
    assert clf_inputs.size() == input_ids.size()
    assert at_labels.size() == input_ids.size()
    return clf_inputs, at_labels


def get_f1(orig, pred, ignore_label=2):
    # if pred.size(-1)!= 1:
    #     pred= get_pre(pred)
    orig = orig.detach().cpu().clone().view(-1)
    pred = pred.detach().cpu().clone().view(-1)
    mask = orig == ignore_label
    pred[mask] = ignore_label
    pred = pred[pred != ignore_label]
    orig = orig[orig != ignore_label]
    assert pred.size() == orig.size()
    return f1_score(orig, pred)


def ceLoss(logits, labels, ignore_idx=None, reduce=True):
    log_size = logits.size()
    lab_size = labels.size()
    logits = logits.view(-1, log_size[-1])
    labels = labels.view(-1)
    loss = F.cross_entropy(
        logits,
        labels,
        size_average=False,
        ignore_index=ignore_idx,
        reduce=reduce,
    )
    return loss


def get_pre(pre):
    pre_size = pre.size()
    pre = pre.view(-1, pre_size[-1])
    pre = pre.argmax(dim=1)
    pre = pre.view(pre_size[0:-1])
    return pre


def get_ceLoss_pre(logits, labels, ignore_idx=None, reduce=True):
    return (
        ceLoss(logits=logits, labels=labels, ignore_idx=ignore_idx, reduce=reduce),
        get_pre(logits),
    )


class InputExample:
    def __init__(self):
        super().__init__()
        self.text = None
        self.tokens = None
        self.input_id = None
        self.input_mask = None
        self.segment_id = None
        self.target_pairs = None
        self.span_labels = None
        self.target_spans = None
        self.lm_sentence = None
        self.clf_sentence = None
        self.offsets = None
        self.orig_len = None
        self.labels = None
        # self.max_seq_len


def count_lines(filePath):
    count = 0
    with open(filePath, "r") as f:
        for line in f:
            count += 1

    return count


class SpanElectraDataConfig:
    def __init__(
        self,
        inFile=None,
        mask_id=None,
        pad_token=None,
        pad_token_id=None,
        max_seq_len=None,
        max_span_len=None,
        mask_ratio=None,
        occur=None,
        ignore_label=2,
    ):
        super().__init__()
        self.inFile = inFile
        self.mask_id = mask_id
        self.pad_token = pad_token
        self.pad_token_id = pad_token_id
        self.max_seq_len = max_seq_len
        self.max_span_len = max_span_len
        self.mask_ratio = mask_ratio
        self.occur = occur
        self.ignore_label = ignore_label

    @classmethod
    def load_from_json(cls, filePath):
        with open(filePath, "r") as f:
            data = f.read()
        data = json.loads(data)
        x = cls()
        for key in data.keys():
            if hasattr(x, key):
                setattr(x, key, data[key])
        return x


class SpanElectraJointTrainConfig:
    def __init__(
        self,
        gen_hidden_size=128,
        embedding_size=256,
        disc_hidden_size=512,
        use_SBPO=False,
        use_SBGO=False,
        vocab_size=30522,
        max_seq_len=512,
        pad_token_id=2,
        pad_token="[PAD]",
        mask_token="[MASK]",
        mask_id=3,
        lowercase=True,
        dummy_id=0,
        ignore_label=2,
        max_span_len=20,
        mask_ratio=0.2,
        device_ids=[0],
        num_workers=0,
        save_dir="/",
        checkpoint_path=None,
        epochs=2,
        learningRate=4e-5,
        train_batch_size=9,
        valid_batch_size=9,
    ):
        super().__init__()
        self.gen_hidden_size = gen_hidden_size
        self.embedding_size = embedding_size
        self.disc_hidden_size = disc_hidden_size
        self.use_SBPO = use_SBPO
        self.use_SBGO = use_SBGO
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.pad_token_id = pad_token_id
        self.pad_token = pad_token
        self.mask_token = mask_token
        self.mask_id = mask_id
        self.lowercase = lowercase
        self.dummy_id = dummy_id
        self.ignore_label = ignore_label
        self.max_span_len = max_span_len
        self.mask_ratio = mask_ratio
        self.device_ids = [0]
        self.num_workers = 0
        self.save_dir = "/"
        self.checkpoint_path = None
        self.epochs = 2
        self.learningRate = 4e-5
        self.train_batch_size = 9
        self.valid_batch_size = 9
        self.init_gen_disc_config()

    def init_gen_disc_config(self):
        self.gen_config = SpanElectraConfig(
            vocab_size=self.vocab_size,
            embedding_size=self.embedding_size,
            hidden_size=self.gen_hidden_size,
            max_span_len=self.max_span_len,
            max_position_embeddings=self.max_seq_len,
            pad_token_id=self.pad_token_id,
            use_SBO=self.use_SBGO,
            max_seq_len=self.max_seq_len,
        )
        self.disc_config = SpanElectraConfig(
            vocab_size=self.vocab_size,
            embedding_size=self.embedding_size,
            hidden_size=self.disc_hidden_size,
            max_span_len=self.max_span_len,
            max_position_embeddings=self.max_seq_len,
            pad_token_id=self.pad_token_id,
            use_SBO=self.use_SBPO,
            max_seq_len=self.max_seq_len,
        )

    @classmethod
    def load_from_json(cls, filePath):
        with open(filePath, "r") as f:
            data = f.read()
        data = json.loads(data)
        x = cls()
        for key in data.keys():
            if hasattr(x, key):
                setattr(x, key, data[key])
        x.init_gen_disc_config()
        return x


def jt_arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_file",
        default="/configs/default.json",
        help="path to config.json file containig training related arguments",
    )
    parser.add_argument(
        "--train_file",
        default=None,
        help="path to feature file containg training data",
    )
    parser.add_argument(
        "--valid_file",
        default=None,
        help="path to feature file containg training data",
    )
    parser.add_argument(
        "--train_occur",
        default=None,
        type=int,
        help="only if you don't want to use whole trainig data then specify number of datapoints you want to use",
    )
    parser.add_argument(
        "--valid_occur",
        default=None,
        type=int,
        help="only if you don't want to use whole validation data then specify number of datapoints you want to use",
    )
    parser.add_argument(
        "--out_dir",
        default="/",
        help="output directory to store model outputs",
    )
    parser.add_argument(
        "--checkpoint_path",
        default=None,
        help="load a model from this check point",
    )
    parser.add_argument("--workers", default=30, type=int, help="number of workers")
    parser.add_argument("--epochs", default=1, type=int, help="number of epochs to run")
    parser.add_argument(
        "--train_batch_size",
        default=8,
        type=int,
        help="training batch size",
    )
    parser.add_argument(
        "--valid_batch_size",
        default=8,
        type=int,
        help="validation batch size",
    )
    parser.add_argument("--lr", default=4e-5, type=float, help="learning rate of model")
    parser.add_argument(
        "--device_ids",
        default=0,
        type=int,
        nargs="+",
        help="list ids of GPU device to use, mention multiple devices for multi GPU",
    )
    parser.add_argument(
        "--log_steps",
        default=25,
        type=int,
        help="logging_step",
    )
    parser.add_argument(
        "--embedding_path",
        default=None,
        help="path to pre trained emebdding",
    )
    return parser


def get_pre_from_span_level_logits(logits, pairs, dummy_id, max_span_len, max_seq_len):
    """
    get prediction of bs, msl from span prediction for SBO
    """
    device = pairs.device
    bs, mx_pair, _ = pairs.size()
    logit_size = logits.size(-1)
    logits = logits.view(bs, mx_pair, max_span_len, logit_size)
    assert logit_size == logits.size(-1)
    pred = torch.zeros((bs, max_seq_len, logit_size), dtype=logits.dtype, device=device)

    for i in range(bs):
        for j in range(mx_pair):
            s, e = pairs[i][j][0], pairs[i][j][1]
            if s == e and s == dummy_id:  # got dummy input
                break
            # print("dskfdhj",pred[i][s:e+1].size(),logits[i][j][0 : e - s + 1].size() , s, e)
            pred[i][s : e + 1] = logits[i][j][0 : e - s + 1]
    return pred


def get_flat_acc(orig, pred, ignore_label=2):
    orig = orig.view(-1)
    pred = pred.view(-1)
    mask = orig != ignore_label
    orig = orig[mask]
    pred = pred[mask]
    count = orig == pred
    return orig[count].size(0) / orig.size(0)
