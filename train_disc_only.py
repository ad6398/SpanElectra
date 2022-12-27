## all traing code, from genrator to discrimnator
## for training genrator
## for creating feature of fake token for discrimnator
import logging
import os
import random
import time
import json
import numpy as np
import pandas as pd
import tokenizers
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AdamW, get_linear_schedule_with_warmup


from configuration_span_electra import SpanElectraConfig
from modelling_span_electra import SpanElectraDiscrimnator
from processing import InputExample, DiscBIDataset
from utilis import (
    get_pre,
    get_f1,
    jt_arg_parse,
    SpanElectraDataConfig,
    SpanElectraJointTrainConfig,
    get_flat_acc,
)

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        # self.record = []

    def update(self, val, n=1):
        self.val = val
        # self.record.append(val)
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def predict(evalData, batch_size, device, model, ignore_label, use_SBO=False, worker=0):
    model.eval()
    evalDataLoader = DataLoader(
        evalData,
        batch_size=batch_size,
        num_workers=worker,
        collate_fn=evalData.collate_fun,
    )

    tdl = tqdm(evalDataLoader, total=len(evalDataLoader))
    total_acc = AverageMeter()
    total_loss = AverageMeter()
    t0 = time.time()
    for idx, batch in enumerate(tdl):

        ids = batch["input_id"].to(device, dtype=torch.long)
        mask_ids = batch["input_mask"].to(device, dtype=torch.long)
        seg_ids = batch["segment_id"].to(device, dtype=torch.long)
        pairs = batch["pairs"].to(device, dtype=torch.long)
        # spans = batch["spans"].to(device, dtype=torch.long)
        labels = batch["labels"].to(device, dtype=torch.long)
        with torch.no_grad():
            logits = model(
                input_ids=ids,
                attention_mask=mask_ids,
                token_type_ids=seg_ids,
                pairs=pairs,
                labels=labels,
            )

        accu = torch.mean(logits[2]).item()

        total_acc.update(accu)

        tdl.set_postfix(accu=total_acc.avg)
    logger.info("validataion acc: {:.4f}".format(total_acc.avg))
    logger.info("validation took {:.2f} sec".format(time.time() - t0))
    return total_acc.avg


def train(
    trainData,
    validData,
    device,
    train_config,
    use_multi_gpu=False,
    device_ids=[],
    log_steps=1,
    pretrained_embedd_path=None,
):

    seed_val = 42
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)
    logger.info("seed value: {} ".format(seed_val))

    model = SpanElectraDiscrimnator(train_config.disc_config)
    if torch.cuda.device_count() > 1 and use_multi_gpu:
        model = nn.DataParallel(model, device_ids=device_ids)
    logger.info(
        "config of model for train {}".format(train_config.disc_config.__dict__)
    )
    start_time = time.time()
    trainDataloader = DataLoader(
        trainData,
        batch_size=train_config.train_batch_size,
        num_workers=train_config.num_workers,
        collate_fn=trainData.collate_fun,
    )
    param_optimizer = list(model.named_parameters())  # get parameter of models
    no_decay = [
        "bias",
        "LayerNorm.bias",
        "LayerNorm.weight",
    ]  ##doubt layers to be not decayed #issue
    optimizer_parameters = [
        {
            "params": [
                p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.001,
        },
        {
            "params": [
                p for n, p in param_optimizer if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_parameters, lr=train_config.learningRate)
    total_len = trainData.__len__()
    logger.info("optimizer: {}".format(optimizer))
    num_steps = total_len / train_config.train_batch_size * train_config.epochs
    logger.info("total steps: {}".format(num_steps))
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=num_steps
    )

    if pretrained_embedd_path is not None:
        logger.info("initializing pre trained embedding")
        embedd_weight = torch.load(pretrained_embedd_path)
        print(type(embedd_weight))
        model.set_input_embedding(embedd_weight)
    model.to(device)
    logger.info("using device: {}".format(device))
    logger.info("################# training started ##################")
    start_time = time.time()
    stats_writer = open(train_config.save_dir + "only_disc_train_stats.txt", "w")
    for epoch_i in range(0, train_config.epochs):
        print("")
        print(
            "======== Epoch {:} / {:} ========".format(epoch_i + 1, train_config.epochs)
        )
        print("Training...")
        t0 = time.time()
        total_loss = AverageMeter()
        total_acc = AverageMeter()
        logger.info(
            "============= Epoch {:} / {:} ===========".format(
                epoch_i + 1, train_config.epochs
            )
        )
        tdl = tqdm(trainDataloader, total=len(trainDataloader))

        model.train()

        for idx, batch in enumerate(tdl):
            tb = time.time()
            ids = batch["input_id"].to(device, dtype=torch.long)
            mask_ids = batch["input_mask"].to(device, dtype=torch.long)
            seg_ids = batch["segment_id"].to(device, dtype=torch.long)
            # lm_sentence = batch["lm_sentence"].to(device, dtype=torch.long)
            pairs = batch["pairs"].to(device, dtype=torch.long)
            labels = batch["labels"].to(device, dtype=torch.long)
            model.zero_grad()
            # at loss, sbo loss, disc f1
            t1 = time.time()
            logits = model(
                input_ids=ids,
                attention_mask=mask_ids,
                token_type_ids=seg_ids,
                pairs=pairs,
                labels=labels,
            )
            t2 = time.time()
            # print(logits.size())
            satat_dict = {}
            at_loss = torch.sum(logits[0])
            satat_dict["at_loss"] = at_loss.item()
            loss = at_loss

            if train_config.disc_config.use_SBO:
                sbo_loss = torch.sum(logits[1])
                satat_dict["sbo_loss"] = sbo_loss.item()
                loss += sbo_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            t3 = time.time()
            disc_f1 = torch.mean(logits[2]).item()
            t4 = time.time()
            satat_dict["loss"] = loss.item()
            satat_dict["f1"] = disc_f1
            satat_dict["epoch"] = epoch_i
            satat_dict["step"] = idx
            total_loss.update(loss.item())
            total_acc.update(disc_f1)

            if idx % log_steps == 0:
                stats_writer.write(json.dumps(satat_dict) + "\n")
            tdl.set_postfix(loss=total_loss.avg, f1_score=total_acc.avg)
            t5 = time.time()
            # print("time for cpu to gpu {}, logits {} , backpop {} , accu cal{} stats write {}".format(t1-tb, t2-t1, t3-t2, t4-t3, t5-t4))

        logger.info("epoch {} took {:.2f} seconds".format(epoch_i, time.time() - t0))
        if validData:
            logger.info("##########validating after epoch end#############")
            vacc = predict(
                validData,
                train_config.valid_batch_size,
                device,
                model,
                ignore_label=train_config.ignore_label,
                use_SBO=train_config.disc_config.use_SBO,
            )

        logger.info(
            "weight and model are saved in dir {}".format(train_config.save_dir)
        )
        torch.save(
            model, train_config.save_dir + "only_disc_model_{}".format(epoch_i)
        )  # save whole model after epoch
        torch.save(
            model.state_dict(),
            train_config.save_dir + "only_disc_model_wight{}".format(epoch_i) + ".pt",
        )  # save weight too to initalize discrimnaotr

    logger.info("total train time {:.2f}".format(time.time() - start_time))
    stats_writer.close()
    # save loss and accu, per step and epoch, may be needed in future


def main():
    parser = jt_arg_parse()
    args = parser.parse_args()

    train_data_arg = SpanElectraDataConfig.load_from_json(args.config_file)
    train_data_arg.inFile = args.train_file
    train_data_arg.occur = args.train_occur

    valid_data_arg = SpanElectraDataConfig.load_from_json(args.config_file)
    valid_data_arg.inFile = args.valid_file
    valid_data_arg.occur = args.valid_occur

    logger.addHandler(
        logging.FileHandler(os.path.join(args.out_dir, "only_disc_model_log.log"), "w")
    )  # initalize logger
    logger.info("training data args")
    logger.info(train_data_arg.__dict__)  # log train data arg
    logger.info("validation data args")
    logger.info(valid_data_arg.__dict__)

    train_config = SpanElectraJointTrainConfig.load_from_json(args.config_file)
    train_config.save_dir = args.out_dir
    train_config.num_workers = args.workers
    train_config.train_batch_size = args.train_batch_size
    train_config.valid_batch_size = args.valid_batch_size
    train_config.learningRate = args.lr
    train_config.checkpoint_path = args.checkpoint_path
    train_config.epochs = args.epochs
    # device_ids = JointTrainingConfig.device_ids
    device_ids = args.device_ids
    if type(device_ids) != list:
        device_ids = [device_ids]
    print(device_ids)
    use_multi_gpu = False
    if len(device_ids) > 1:
        use_multi_gpu = True
    if torch.cuda.is_available():
        device = torch.device(
            "cuda:" + str(device_ids[0])
        )  # use first device as main device
        print("There are %d GPU(s) available." % torch.cuda.device_count())
        # print("We will use the GPU:", torch.cuda.get_device_name(device_ids[0]))
    else:
        print("No GPU available, using the CPU instead.")
        device = torch.device("cpu")
    trainData = DiscBIDataset(train_data_arg, 16)
    validData = DiscBIDataset(valid_data_arg, 16)
    # otarg= Joint_trainDataArgs()
    # ovarg= Joint_validDataArgs()
    # trainData = MLMSpanElectraDataset(otarg)
    # validData = MLMSpanElectraDataset(ovarg)

    train(
        trainData=trainData,
        validData=validData,
        device=device,
        train_config=train_config,
        use_multi_gpu=use_multi_gpu,
        device_ids=device_ids,
        log_steps=args.log_steps,
        pretrained_embedd_path=args.embedding_path,
    )
    # torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

# python train_disc_only.py --config_file "/home/amardeep/spanElectra/keyword-language-modeling/configs/od_bert_base_config.json" --train_file "/media/data_dump/Amardeep/test_fol/od_bert/train.bin" --valid_file "/media/data_dump/Amardeep/test_fol/od_bert/valid.bin" --out_dir "/media/data_dump/Amardeep/test_fol/od_bert/" --workers 0 --epochs 1 --lr 3e-5 --device_ids 1
