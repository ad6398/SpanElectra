import os
import torch
from tqdm import tqdm
import pandas as pd
import torch.nn as nn
import numpy as np
import time, random
import logging
from processing import SpanELectraDataset, InputExample
from configuration_span_electra import SpanElectraConfig

from modelling_span_electra import SpanElectraForPretraining
from argument import SE_trainDataArgs, SE_validDataArgs, SE_trainingConfig

from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader
import tokenizers
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup
from utilis import plot2, save_stats, get_f1
from sklearn.metrics import f1_score
from matplotlib import pyplot as plt

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


def get_flat_acc_t(orig, pred, ignore_label=2):
    tl = 0
    totalScore = 0
    for x, y in zip(orig, pred):
        if x != ignore_label:
            tl += 1
            if x == y:
                totalScore += 1
    if tl == 0:
        return 0, tl

    return totalScore / tl, tl


def predictSE(evalData, batch_size, device, model, ignore_label, worker=0):
    model.eval()

    evalDataLoader = DataLoader(
        evalData,
        batch_size=batch_size,
        num_workers=worker,
        collate_fn=evalData.collate_fun,
    )

    tdl = tqdm(evalDataLoader, total=len(evalDataLoader))
    sboLoss = AverageMeter()
    sboF1 = AverageMeter()

    atLoss = AverageMeter()
    atF1 = AverageMeter()
    predictions = []
    t0 = time.time()
    for idx, batch in enumerate(tdl):

        # ids= batch['input_id'].to(device, dtype= torch.long)
        mask_ids = batch["input_mask"].to(device, dtype=torch.long)
        seg_ids = batch["segment_id"].to(device, dtype=torch.long)
        lm_sentence = batch["clf_sentence"].to(device, dtype=torch.long)
        pairs = batch["pairs"].to(device, dtype=torch.long)
        span_labels = batch["span_labels"].to(device, dtype=torch.long)
        all_token_labels = batch["all_token_labels"].to(device, dtype=torch.long)
        with torch.no_grad():
            at_loss, at_logits, sbo_loss, sbo_logits = model(
                input_ids=lm_sentence,
                attention_mask=mask_ids,
                token_type_ids=seg_ids,
                pairs=pairs,
                span_labels=span_labels,
                all_tok_labels=all_token_labels,
            )
        if at_loss is not None:
            at_logits = at_logits.view(-1, at_logits.size(-1))
            at_pred = at_logits.argmax(dim=1)
            all_token_labels = all_token_labels.view(-1)
            at_f1 = get_f1(all_token_labels, at_pred, ignore_label=ignore_label)
            atLoss.update(at_loss.item(), batch_size)
            atF1.update(at_f1, batch_size)

        if sbo_loss is not None:
            clf_logits = sbo_logits.view(-1, sbo_logits.size(-1))
            sbo_pred = clf_logits.argmax(dim=1)
            span_labels = span_labels.view(-1)
            sbo_f1 = get_f1(span_labels, sbo_pred, ignore_label=ignore_label)
            sboLoss.update(sbo_loss.item(), batch_size)
            sboF1.update(sbo_f1, batch_size)

        tdl.set_postfix(
            sbo_loss=sboLoss.avg, sbo_f1=sboF1.avg, at_loss=atLoss.avg, at_f1=atF1.avg
        )

    logger.info(
        "Validation: sbo_loss {:.4f}, sbo_f1 {:.4f}, all_token_loss: {:.4f}, all_token_f1 {:.4f}  ".format(
            sboLoss.avg, sboF1.avg, atLoss.avg, atF1.avg
        )
    )
    logger.info("validation took {:.2f} sec".format(time.time() - t0))

    return sboLoss.avg, sboF1.avg, atLoss.avg, atF1.avg


def trainSE(trainData, validData, device, train_config):

    seed_val = 42
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)
    logger.info("seed value: {} ".format(seed_val))

    model = SpanElectraForPretraining(train_config.modelConfig)

    logger.info(
        "config of model for train {}".format(train_config.modelConfig.__dict__)
    )
    logger.info("preaparing train data in batches")
    start_time = time.time()
    batch_size = train_config.train_batch_size
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
    logger.info("using device: {}".format(device))
    logger.info("################# training started ##################")
    logger.info("initializing weight of disc with gen")
    model.load_state_dict(torch.load(train_config.generator_weight_path), strict=False)
    logger.info("weight loaded")
    start_time = time.time()
    epoch_sbo_loss = []
    epoch_sbo_accc = []
    batch_sbo_loss = []
    batch_sbo_accc = []
    valid_sbo_loss = []
    valid_sbo_accc = []

    epoch_at_loss = []
    epoch_at_accc = []
    batch_at_loss = []
    batch_at_accc = []
    valid_at_loss = []
    valid_at_accc = []

    for epoch_i in range(0, train_config.epochs):
        print("")
        print(
            "======== Epoch {:} / {:} ========".format(epoch_i + 1, train_config.epochs)
        )
        print("Training...")
        t0 = time.time()

        sboLoss = AverageMeter()
        sboF1 = AverageMeter()

        atLoss = AverageMeter()
        atF1 = AverageMeter()

        logger.info(
            "============= Epoch {:} / {:} ===========".format(
                epoch_i + 1, train_config.epochs
            )
        )
        tdl = tqdm(trainDataloader, total=len(trainDataloader))

        model.train()

        for idx, batch in enumerate(tdl):
            tb = time.time()
            # ids= batch['input_id'].to(device, dtype= torch.long)
            mask_ids = batch["input_mask"].to(device, dtype=torch.long)
            seg_ids = batch["segment_id"].to(device, dtype=torch.long)
            clf_sentence = batch["clf_sentence"].to(device, dtype=torch.long)
            pairs = batch["pairs"].to(device, dtype=torch.long)
            span_labels = batch["span_labels"].to(device, dtype=torch.long)
            all_token_labels = batch["all_token_labels"].to(device, dtype=torch.long)

            model.zero_grad()

            at_loss, at_logits, sbo_loss, sbo_logits = model(
                input_ids=clf_sentence,
                attention_mask=mask_ids,
                token_type_ids=seg_ids,
                pairs=pairs,
                span_labels=span_labels,
                all_tok_labels=all_token_labels,
            )
            loss = None
            if at_loss is not None:
                loss = at_loss
                at_logits = at_logits.view(-1, at_logits.size(-1))
                at_pred = at_logits.argmax(dim=1)
                all_token_labels = all_token_labels.view(-1)
                at_f1 = get_f1(
                    all_token_labels, at_pred, ignore_label=train_config.ignore_label
                )
                atLoss.update(at_loss.item(), batch_size)
                atF1.update(at_f1, batch_size)
                batch_at_loss.append(at_loss.item())
                batch_at_accc.append(at_f1)

            if sbo_loss is not None:
                loss += sbo_loss
                clf_logits = sbo_logits.view(-1, sbo_logits.size(-1))
                sbo_pred = clf_logits.argmax(dim=1)
                span_labels = span_labels.view(-1)
                sbo_f1 = get_f1(
                    span_labels, sbo_pred, ignore_label=train_config.ignore_label
                )
                sboLoss.update(sbo_loss.item(), batch_size)
                sboF1.update(sbo_f1, batch_size)
                batch_sbo_loss.append(sbo_loss.item())
                batch_sbo_accc.append(sbo_f1)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            if idx % 50 == 0:
                logger.info(
                    "\nepoch {}, batch no {}: sbo_loss {:.4f}, sbo_f1 {:.4f}, all_token_loss: {:.4f}, all_token_f1 {:.4f}  ".format(
                        epoch_i, idx, sboLoss.avg, sboF1.avg, atLoss.avg, atF1.avg
                    )
                )
            tdl.set_postfix(
                sbo_loss=sboLoss.avg,
                sbo_f1=sboF1.avg,
                at_loss=atLoss.avg,
                at_f1=atF1.avg,
            )

        logger.info("epoch {} took {:.2f} seconds".format(epoch_i, time.time() - t0))
        if validData:
            logger.info("##########validating after epoch end#############")
            vsl, vsa, vatl, vata = predictSE(
                validData,
                train_config.valid_batch_size,
                device,
                model,
                ignore_label=train_config.ignore_label,
                worker=train_config.worker,
            )
            valid_at_loss.append(vatl)
            valid_at_accc.append(vata)
            valid_sbo_loss.append(vsl)
            valid_sbo_accc.append(vsa)

        epoch_at_loss.append(atLoss.avg)
        epoch_at_accc.append(atF1.avg)
        epoch_sbo_loss.append(sboLoss.avg)
        epoch_sbo_accc.append(sboF1.avg)

        logger.info(
            "weight stats, and model are saved in dir {}".format(train_config.save_dir)
        )
        torch.save(
            model, train_config.save_dir + "SEmodel_{}".format(epoch_i)
        )  # save whole model after epoch
        torch.save(
            model.state_dict(),
            train_config.save_dir + "SEmodel_wight{}".format(epoch_i) + ".pt",
        )  # save weight

    logger.info("total train time {:.2f}".format(time.time() - start_time))

    # save loss and accu, per step and epoch, may be needed in future
    save_stats(
        save_dir=train_config.save_dir,
        name="twoStageDisc",
        epoch_train_sbo_loss=epoch_sbo_loss,
        epoch_train_sbo_acc=epoch_sbo_accc,
        epoch_train_at_loss=epoch_at_loss,
        epoch_train_at_acc=epoch_at_accc,
        batch_train_sbo_loss=batch_sbo_loss,
        batch_train_sbo_acc=batch_sbo_accc,
        batch_train_at_loss=batch_at_loss,
        batch_train_at_acc=batch_at_accc,
        epoch_valid_sbo_loss=valid_sbo_loss,
        epoch_valid_sbo_acc=valid_sbo_accc,
        epoch_valid_at_loss=valid_at_loss,
        epoch_valid_at_acc=valid_at_accc,
    )

    # loss and accu per epoch plot for valid and train
    plot2(
        ts=num_steps,
        loss={"train": batch_at_loss},
        acc={"train": batch_at_accc},
        x1="batch num",
        y1="all token loss",
        x2="batch num",
        y2="all tok f1",
        figName="disc_batch_all_tok",
        dire=train_config.save_dir,
    )
    # plot sbo loss , acc per batch
    plot2(
        ts=num_steps,
        loss={"train": batch_sbo_loss},
        acc={"train": batch_sbo_accc},
        x1="batch num",
        y1="sbo loss",
        x2="batch num",
        y2="sbo f1 score",
        figName="disc_batch_sbo",
        dire=train_config.save_dir,
    )


def main():
    device_id = SE_trainingConfig.device_id
    if torch.cuda.is_available():
        device = torch.device("cuda:" + str(device_id))
        print("There are %d GPU(s) available." % torch.cuda.device_count())
        print("We will use the GPU:", torch.cuda.get_device_name(device_id))
    else:
        print("No GPU available, using the CPU instead.")
        device = torch.device("cpu")
    targ = SE_trainDataArgs()
    varg = SE_validDataArgs()
    logger.addHandler(
        logging.FileHandler(os.path.join(targ.out_dir, "disc_logFile.log"), "w")
    )  # initalize logger
    logger.info("training data args")
    logger.info(targ.__dict__)  # log train data arg
    logger.info("validation data args")
    logger.info(varg.__dict__)
    trainData = SpanELectraDataset(targ)
    validData = SpanELectraDataset(varg)
    trainSE(
        trainData=trainData,
        validData=validData,
        device=device,
        train_config=SE_trainingConfig,
    )


if __name__ == "__main__":
    main()
