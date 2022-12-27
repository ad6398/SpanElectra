## all masking schemese: span masking, token masking

# from processing import InputExample
import math

import numpy as np
import torch

from utilis import get_disc_input_at_labels, get_mlm_loss_out_sbo_labels


def merge_intervals(intervals):
    intervals = sorted(intervals, key=lambda x: x[0])
    merged = []
    for interval in intervals:
        # if the list of merged intervals is empty or if the current
        # interval does not overlap with the previous, simply append it.
        if not merged or merged[-1][1] + 1 < interval[0]:
            merged.append(interval)
        else:
            # otherwise, there is overlap, so we merge the current and previous
            # intervals.
            merged[-1][1] = max(merged[-1][1], interval[1])

    return merged


def get_span(pairs, ids, pad_token_id, max_span_len):
    spans = []
    for s, e in pairs:
        cspan = []
        for idx in range(s, e + 1):
            cspan.append(ids[idx])
        if len(cspan) > max_span_len:
            cspan = cspan[:max_span_len]
        cspan = pad_to_len(cspan, pad_token_id, max_span_len)
        spans.append(cspan)
    return spans


def trim_spans(spans, max_span_len):
    nw_span = []
    for s, e in spans:
        if e - s + 1 > max_span_len:
            e = s + max_span_len - 1
        nw_span.append([s, e])
    return nw_span


def get_span_masked_feature(
    example,
    mask_id,
    max_seq_len,
    geometric_dist=0.2,
    max_span_len=20,
    mask_ratio=0.2,
    span_lower=1,
    span_upper=10,
    pad_token="[PAD]",
    pad_token_id=2,
):
    """return Input example with lm_sentence , target_pairs: [start, end] of each apirs,
    targets_span: token_ids of each span
    """

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

    sent_length = example.orig_len
    example.input_id = pad_to_len(example.input_id, pad_token_id, max_seq_len)
    example.input_mask = [1] * (sent_length + 2)
    example.input_mask = pad_to_len(example.input_mask, 0, max_seq_len)  # 0 for padded
    example.segment_id = []
    example.segment_id = pad_to_len(
        example.segment_id, 1, max_seq_len
    )  # 1 as these are single sentence
    mask_num = math.ceil(sent_length * mask_ratio)
    mask = set()
    total_len = len(example.input_id)

    target_pairs = []
    target_spans = []

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
    ids = example.input_id[:]

    target_pairs = merge_intervals(target_pairs)  # merge colliding interval
    assert len(mask) == sum([e - s + 1 for s, e in target_pairs])

    target_pairs = trim_spans(target_pairs, max_span_len)
    # target_spans = get_span(target_pairs, ids, pad_token_id, max_span_len)

    lm_sent, lm_labels = mask_sent(
        example.input_id, target_pairs, mask_id, mask, pad_token=pad_token_id
    )

    assert len(lm_sent) == len(example.input_id)
    # print(len(lm_labels))
    assert len(lm_labels) == len(example.input_id)
    assert len(target_pairs) != 0
    # assert len(target_spans) == len(target_pairs)
    # assert len(target_spans[0]) == max_span_len

    example.target_pairs = target_pairs
    example.target_spans = target_spans
    example.lm_sentence = lm_sent
    example.labels = lm_labels

    return example


def mask_sent(ids, spans, mask_id, mask_set, pad_token=2):
    lm_sent = ids[:]
    lm_labels = [pad_token] * len(ids)
    for s, e in spans:
        for idx in range(s, e + 1):
            assert idx in mask_set
            lm_sent[idx] = ids[idx]
            lm_labels[idx] = ids[idx]
            rand = np.random.random()
            if rand < 0.8:
                lm_sent[idx] = mask_id
            elif rand < 0.9:
                # we can avoid sep or clf token selection by ids[1:-1]
                lm_sent[idx] = np.random.choice(ids)

    return lm_sent, lm_labels


def pad_to_len(list, pad, length):
    """pad a single list to len"""
    cur_len = len(list)
    if cur_len > length:
        raise ValueError("len of list is greater than padding length")

    req_len = length - cur_len

    list = list + [pad] * req_len
    assert len(list) == length
    return list


def get_fake_batch(
    batch, generator, max_span_len, vocab_size, pad_token_id, max_seq_len
):
    """convert masked span to cl"""
    dummy_id = 0
    assert type(batch) is not None
    bs = len(batch)
    pairs = [x["pairs"] for x in batch]
    spans = [x["spans"] for x in batch]
    input_ids = [x["input_id"] for x in batch]
    imask = [x["input_mask"] for x in batch]
    sid = [x["segment_id"] for x in batch]
    ori_len = [x["orig_len"] for x in batch]
    lm_sent = [x["lm_sentence"] for x in batch]
    assert len(input_ids) == bs
    assert len(lm_sent) == bs
    mx_pairs = max(len(x) for x in pairs)
    for i, (s, p) in enumerate(zip(spans, pairs)):
        # dummy_id is to generate fake pairs to satisfy equal lenght
        pairs[i] = pad_to_len(p, [dummy_id, dummy_id], mx_pairs)
        #         batch[i]['pairs']= pairs[i]
        spans[i] = pad_to_len(s, [pad_token_id] * max_span_len, mx_pairs)
    #         batch[i]['spans']= spans[i]

    gen_batch = {
        # 'tokens': torch.tensor(curr_ex.tokens, dtype = torch.long),
        "input_id": torch.tensor(input_ids, dtype=torch.long),
        "input_mask": torch.tensor(imask, dtype=torch.long),
        # 'offsets': torch.tensor(batch['offsets'], dtype = torch.long),
        "segment_id": torch.tensor(sid, dtype=torch.long),
        "orig_len": torch.tensor(ori_len, dtype=torch.long),
        "lm_sentence": torch.tensor(lm_sent, dtype=torch.long),
        "pairs": torch.tensor(pairs, dtype=torch.long),
        "spans": torch.tensor(spans, dtype=torch.long),
    }

    model = torch.load(generator)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    input_ids = gen_batch["input_id"].to(device, dtype=torch.long)
    mask_ids = gen_batch["input_mask"].to(device, dtype=torch.long)
    seg_ids = gen_batch["segment_id"].to(device, dtype=torch.long)
    lm_sentence = gen_batch["lm_sentence"].to(device, dtype=torch.long)
    pairs = gen_batch["pairs"].to(device, dtype=torch.long)
    spans = gen_batch["spans"].to(device, dtype=torch.long)
    with torch.no_grad():
        loss, logits = model(
            input_ids=lm_sentence,
            attention_mask=mask_ids,
            token_type_ids=seg_ids,
            pairs=pairs,
            span_labels=spans,
        )

    mlm_loss, gen_tokens, span_labels = get_mlm_loss_out_sbo_labels(
        logits, labels=spans, ignore_labels=pad_token_id, pad_token_id=pad_token_id
    )

    assert gen_tokens.size() == spans.size()
    assert span_labels.size() == spans.size()

    clf_inputs, all_tok_labels = get_disc_input_at_labels(
        input_ids,
        gen_tokens=gen_tokens,
        span_labels=span_labels,
        pairs=pairs,
        dummy_id=dummy_id,
        pad_token_id=pad_token_id,
        ignore_label=pad_token_id,
    )  # ignore label

    # print(clf_sentences[2],"\n", input_ids[2],"\n", lm_sent[2])
    new_batch = {
        # 'tokens': torch.tensor(curr_ex.tokens, dtype = torch.long),
        "input_id": torch.tensor(input_ids, dtype=torch.long),
        "input_mask": torch.tensor(imask, dtype=torch.long),
        # 'offsets': torch.tensor(batch['offsets'], dtype = torch.long),
        "segment_id": torch.tensor(sid, dtype=torch.long),
        "orig_len": torch.tensor(ori_len, dtype=torch.long),
        "clf_sentence": torch.tensor(clf_inputs, dtype=torch.long),
        "pairs": torch.tensor(pairs, dtype=torch.long),
        "spans": torch.tensor(spans, dtype=torch.long),
        "span_labels": torch.tensor(span_labels, dtype=torch.long),
        "all_token_labels": torch.tensor(all_tok_labels, dtype=torch.long),
    }
    #     print(new_batch)
    return new_batch
