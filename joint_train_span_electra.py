import logging
import os
import random
import time

import numpy as np
import pandas as pd
import tokenizers
import torch
import torch.nn as nn
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AdamW, get_linear_schedule_with_warmup
import json
from argument import Joint_trainDataArgs, Joint_validDataArgs, JointTrainingConfig
from modelling_span_electra import SpanaElectraJoint
from processing import (
    InputExample,
    MLMSpanElectraDataset,
    SpanELectraLazyDataset,
    CachedBinaryIndexedDataset,
)
from utilis import (
    plot2,
    save_stats,
    SpanElectraJointTrainConfig,
    SpanElectraDataConfig,
    jt_arg_parse,
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
        # self.record= []

    def update(self, val, n=1):
        self.val = val
        # self.record.append(val)
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def predict(
    evalData,
    batch_size,
    device,
    model,
    ignore_label,
    worker=0,
    use_multi_gpu=False,
    device_ids=[],
):
    model.eval()

    evalDataLoader = DataLoader(
        evalData,
        batch_size=batch_size,
        num_workers=worker,
        collate_fn=evalData.collate_fun,
    )

    tdl = tqdm(evalDataLoader, total=len(evalDataLoader))
    genLoss = AverageMeter()
    genAcc = AverageMeter()
    discLoss = AverageMeter()
    discF1 = AverageMeter()

    t0 = time.time()
    for idx, batch in enumerate(tdl):

        # ids= batch['input_id'].to(device, dtype= torch.long)
        mask_ids = batch["input_mask"].to(device, dtype=torch.long)
        seg_ids = batch["segment_id"].to(device, dtype=torch.long)
        lm_sentence = batch["lm_sentence"].to(device, dtype=torch.long)
        pairs = batch["pairs"].to(device, dtype=torch.long)
        # spans = batch["spans"].to(device, dtype=torch.long)
        labels = batch["labels"].to(device, dtype=torch.long)

        with torch.no_grad():
            logits = model(
                input_ids=lm_sentence,
                attention_mask=mask_ids,
                token_type_ids=seg_ids,
                pairs=pairs,
                labels=labels,
            )

        disc_f1 = torch.sum(logits[5]).item()
        gen_accu = torch.sum(logits[2]).item()
        discF1.update(disc_f1)
        genAcc.update(gen_accu)

        tdl.set_postfix(
            gen_accu=genAcc.avg,
            disc_f1=discF1.avg,
        )
    logger.info(
        "Validation: generator_accu: {:.4f}, disc_f1 {:.4f}  ".format(
            genAcc.avg, discF1.avg
        )
    )
    logger.info("validation took {:.2f} sec".format(time.time() - t0))

    return genLoss.avg, discF1.avg


def train(
    trainData,
    validData,
    device,
    train_config,
    use_multi_gpu=False,
    device_ids=[],
    log_steps=10,
):

    seed_val = 42
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)
    logger.info("seed value: {} ".format(seed_val))
    batch_size = train_config.train_batch_size

    model = SpanaElectraJoint(train_config.gen_config, train_config.disc_config)
    if torch.cuda.device_count() > 1 and use_multi_gpu:
        model = nn.DataParallel(model, device_ids=device_ids)

    logger.info("generator config {}".format(train_config.gen_config.__dict__))
    logger.info("discrimnator config {}".format(train_config.disc_config.__dict__))
    logger.info("preaparing train data in batches")
    start_time = time.time()

    trainDataloader = DataLoader(
        trainData,
        batch_size=train_config.train_batch_size,
        num_workers=train_config.num_workers,
        collate_fn=trainData.collate_fun,
    )
    logger.info("batching took {:.3f} sec".format(time.time() - start_time))

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

    model.to(device)
    start_epoch = -1
    if train_config.checkpoint_path is not None:
        logger.info("loading model from a check point\n")
        checkPoint = torch.load(train_config.checkpoint_path)
        model.load_state_dict(checkPoint["model_state_dict"])
        start_epoch = checkPoint["epoch"]
        optimizer.load_state_dict(checkPoint["optimizer_state_dict"])
        logger.info("done loading")

    logger.info("using device: {}".format(device))
    logger.info("################# training started ##################")

    stats_writer = open(train_config.save_dir + "joint_train_stats.txt", "w")

    for epoch_i in range(start_epoch + 1, train_config.epochs):
        print("")
        print(
            "======== Epoch {:} / {:} ========".format(epoch_i + 1, train_config.epochs)
        )
        print("Training...")
        t0 = time.time()
        genLoss = AverageMeter()
        genAcc = AverageMeter()

        discLoss = AverageMeter()
        discF1 = AverageMeter()

        logger.info(
            "============= Epoch {:} / {:} ===========".format(
                epoch_i + 1, train_config.epochs
            )
        )

        tdl = tqdm(trainDataloader, total=len(trainDataloader))

        model.train()

        for idx, batch in enumerate(tdl):

            # ids= batch['input_id'].to(device, dtype= torch.long)
            mask_ids = batch["input_mask"].to(device, dtype=torch.long)
            seg_ids = batch["segment_id"].to(device, dtype=torch.long)
            lm_sentence = batch["lm_sentence"].to(device, dtype=torch.long)
            pairs = batch["pairs"].to(device, dtype=torch.long)
            # spans = batch["spans"].to(device, dtype=torch.long)
            labels = batch["labels"].to(device, dtype=torch.long)

            model.zero_grad()

            logits = model(
                input_ids=lm_sentence,
                attention_mask=mask_ids,
                token_type_ids=seg_ids,
                pairs=pairs,
                labels=labels,
            )
            stats_dic = {}
            gen_loss = torch.sum(logits[0])
            disc_loss = torch.sum(logits[3])
            stats_dic["gen_lm_loss"] = gen_loss.item()
            stats_dic["disc_at_loss"] = disc_loss.item()

            if train_config.use_SBGO:
                gen_sbo_loss = torch.sum(logits[1])
                stats_dic["gen_sbo_loss"] = gen_sbo_loss.item()
                gen_loss += gen_sbo_loss

            if train_config.use_SBPO:
                disc_sbo_loss = torch.sum(logits[4])
                stats_dic["disc_sbo_loss"] = disc_sbo_loss.item()
                disc_loss += disc_sbo_loss

            stats_dic["gen_f1"] = torch.sum(logits[2]).item()
            stats_dic["disc_f1"] = torch.sum(logits[5]).item()

            loss = gen_loss + disc_loss  # gen_loss + disc loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            genLoss.update(gen_loss.item())
            genAcc.update(stats_dic["gen_f1"])
            discLoss.update(disc_loss.item())
            discF1.update(stats_dic["disc_f1"])

            if idx % log_steps == 0:
                stats_writer.write(json.dumps(stats_dic) + "\n")

            tdl.set_postfix(
                gen_loss=genLoss.avg,
                gen_accu=genAcc.avg,
                disc_loss=discLoss.avg,
                disc_f1=discF1.avg,
            )

        if validData:
            logger.info("##########validating after epoch end#############")
            valid_gen_accu, valid_disc_f1 = predict(
                validData,
                train_config.valid_batch_size,
                device,
                model,
                ignore_label=train_config.ignore_label,
            )

        torch.save(
            {
                "epoch": epoch_i,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            },
            train_config.save_dir + "jointSE_checkpoint_{}.pt".format(epoch_i),
        )  # save whole model after epoch
        torch.save(
            model.state_dict(),
            train_config.save_dir + "jointSEmodel_weight{}".format(epoch_i) + ".pt",
        )

    stats_writer.close()


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
        logging.FileHandler(os.path.join(args.out_dir, "jointTrain_logFile.log"), "w")
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
    trainData = CachedBinaryIndexedDataset(train_data_arg, 8)
    validData = CachedBinaryIndexedDataset(valid_data_arg, 8)
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
    )
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
    torch.cuda.empty_cache()

# python joint_train_span_electra.py --config_file "/home/amardeep/spanElectra/keyword-language-modeling/configs/default.json" --train_file "/media/data_dump/Amardeep/spanElectra/out/jfeat/train.txt" --valid_file "/media/data_dump/Amardeep/spanElectra/out/jfeat/valid.txt" --out_dir "/media/data_dump/Amardeep/spanElectra/out/jfeat/" --workers 0 --epochs 1 --lr 3e-5 --device_ids 1
