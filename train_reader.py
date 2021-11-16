# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import time
import sys
import torch
import transformers
import numpy as np
import os
import os.path as osp
import copy
import json
import pdb

from tqdm import tqdm

from pathlib import Path
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler, SequentialSampler
from src.options import Options

import nltk.tokenize as tk
import nltk.stem.porter as pt

import src.slurm
import src.util
import src.evaluation
import src.data
import src.model

import warnings
warnings.filterwarnings("ignore")


def train(model, optimizer, scheduler, step, train_dataset, eval_dataset, opt, collator, best_dev_em, checkpoint_path, tokenizer):

    torch.manual_seed(opt.global_rank + opt.seed)  # different seed for different sampling depending on global_rank
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        batch_size=opt.per_gpu_batch_size,
        drop_last=True,
        num_workers=6,
        collate_fn=collator
    )

    loss, curr_loss = 0.0, 0.0
    epoch = 0
    step_per_epoch = int(len(train_dataloader) / opt.per_gpu_batch_size)
    model.train()
    patience = 0

    tk_tokenizer = tk.WordPunctTokenizer()
    pt_stemmer = pt.PorterStemmer()

    # import statistics
    while epoch < opt.epochs:
        epoch += 1
        for i, batch in tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc=f'Training | epoch {epoch}'):
            step += 1
            (idx, labels, _, context_ids, context_mask) = batch
            #  context_ids / context_mask: torch.Size([32, 1, 120])
            # pdb.set_trace()
            train_loss = model(
                input_ids=context_ids.cuda(),
                attention_mask=context_mask.cuda(),
                labels=labels.cuda()
            )[0]

            train_loss.backward()

            # if step % opt.accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), opt.clip)
            optimizer.step()
            scheduler.step()
            model.zero_grad()

            train_loss = src.util.average_main(train_loss, opt)
            curr_loss += train_loss.item()

            del train_loss

        patience += 1
        dev_em = evaluate(model, eval_dataset, tokenizer, collator, opt, tk_tokenizer, pt_stemmer)

        model.train()
        if opt.is_main:
            log = f"epoch {epoch} |"
            log += f"step {step} |"
            log += f"train loss: {curr_loss/step_per_epoch:.3f} |"
            log += f"evaluation: {100*dev_em:.2f}EM |"
            # log += f"stem: {100*dev_stem_em:.2f}EM |"
            log += f"lr: {scheduler.get_last_lr()[0]:.5f}"
            logger.info(log)

            if dev_em > best_dev_em:
                patience = 0
                # model save
                # best_model = copy.deepcopy(model)
                # best_scheduler = copy.deepcopy(scheduler)
                # best_optimizer = copy.deepcopy(optimizer)
                # best_step = step

                src.util.save(model, optimizer, scheduler, step, dev_em, opt, checkpoint_path, 'best_dev')
                best_dev_em = dev_em

            curr_loss = 0

            if patience > opt.early_stop:
                logger.info(f"early stop in epoch {epoch}")
                break

    # logger.info(f"early stop in epoch {epoch}")

    log = f"stop epoch {epoch} |"
    log += f"evaluation: {100*best_dev_em:.2f}EM |"
    logger.info(log)


def evaluate(model, dataset, tokenizer, collator, opt, tk_tokenizer, stemmer):
    sampler = SequentialSampler(dataset)
    dataloader = DataLoader(dataset,
                            sampler=sampler,
                            batch_size=opt.per_gpu_batch_size,
                            drop_last=False,
                            num_workers=10,
                            collate_fn=collator
                            )
    model.eval()
    total = 0
    exactmatch = []
    # stem_exactmatch = []
    # include_score = []
    model = model.module if hasattr(model, "module") else model
    with torch.no_grad():
        for i, batch in tqdm(enumerate(dataloader), total=len(dataloader), desc=f'Eval'):
            (idx, _, _, context_ids, context_mask) = batch

            outputs = model.generate(
                input_ids=context_ids.cuda(),
                attention_mask=context_mask.cuda(),
                max_length=50
            )

            generated_sents = tokenizer.batch_decode(outputs, skip_special_tokens=True)

            for k, ans in enumerate(generated_sents):
                # ans = tokenizer.decode(o, skip_special_tokens=True)
                gold = dataset.get_example(idx[k])['answer']

                # pdb.set_trace()

                score = src.evaluation.ems(ans, gold)
                # include_score = src.evaluation.includ_ems(ans, gold)
                # stem_score = src.evaluation.stem_ems(ans, gold, tk_tokenizer, stemmer)
                exactmatch.append(score)
                # stem_exactmatch.append(stem_score)
                # include_score.append(include_score)

                total += 1

    exactmatch, total = src.util.weighted_average(np.mean(exactmatch), total, opt)
    # stem_exactmatch, total = src.util.weighted_average(np.mean(stem_exactmatch), total, opt)

    # return exactmatch, stem_exactmatch
    return exactmatch


if __name__ == "__main__":
    options = Options()
    options.add_reader_options()
    options.add_optim_options()
    opt = options.parse()
    torch.cuda.set_device(opt.gpu)

    #opt = options.get_options(use_reader=True, use_optim=True)

    torch.manual_seed(opt.seed)
    src.slurm.init_distributed_mode(opt)
    src.slurm.init_signal_handler()
    this_dir = osp.dirname(__file__)
    data_path = osp.abspath(osp.join(this_dir, '..', '..', 'data', 'LaKo'))
    cache_dir = osp.abspath(osp.join(this_dir, '..', '..', 'data', '.cache', 'transformers'))

    opt.device = opt.gpu

    if "with fact" in opt.model_path:
        with_fact = "_with_fact"
    else:
        with_fact = ""

    if opt.model_path == "none":
        from_scratch = "_from_scratch"
    else:
        from_scratch = ""
    if opt.use_fact == "yes":
        fact_para = f"_stream_{opt.stream}_content_{opt.n_context}_"
    else:
        fact_para = ""

    train_data_name = opt.train_data.split('_')[-1]
    if train_data_name[0] == "v":

        iter_name = f"iter_{opt.version}"
    else:
        iter_name = ""
    model_name = f"{opt.dataset}_{opt.model_size}_batch_{opt.per_gpu_batch_size}_maxLen_{opt.text_maxlength}{fact_para}{with_fact}{from_scratch}{iter_name}"

    opt.name = model_name

    checkpoint_path = osp.join(data_path, opt.checkpoint_dir, opt.name)

    os.makedirs(checkpoint_path, exist_ok=True)

    log_dir = osp.join(checkpoint_path, 'run.log')

    logger = src.util.init_logger(
        opt.is_main,
        opt.is_distributed,
        log_dir
    )

    model_name = 't5-' + opt.model_size
    model_class = src.model.FiDT5

    # load data

    tokenizer = transformers.T5Tokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    collator = src.data.Collator(opt.text_maxlength, tokenizer, answer_maxlength=opt.answer_maxlength, stream=opt.stream)

    train_data_path = osp.join(data_path, opt.dataset, opt.train_data)
    eval_data_path = osp.join(data_path, opt.dataset, opt.eval_data)
    with open(train_data_path, 'r') as fin:
        train_examples = json.load(fin)
    train_dataset = src.data.Dataset(train_examples, opt)
    with open(eval_data_path, 'r') as fin:
        eval_examples = json.load(fin)

    eval_dataset = src.data.Dataset(eval_examples, opt)
    t5 = transformers.T5ForConditionalGeneration.from_pretrained(model_name, cache_dir=cache_dir)
    model = src.model.FiDT5(t5.config)

    if opt.model_path == "none":
        logger.info(f"load t5...")
        model.load_t5(t5.state_dict())
        logger.info(f"into cuda...")
        model = model.cuda()

    else:

        opt.model_path = osp.join(data_path, opt.model_path)
        model, optimizer, scheduler, opt_checkpoint, step, best_dev_em = src.util.load(model_class, opt.model_path, opt, reset_params=True)
        logger.info(f"Model loaded from {opt.model_path}")

    model.set_checkpoint(opt.use_checkpoint)

    step_per_epoch = int(len(train_dataset) / opt.per_gpu_batch_size)
    opt.warmup_steps = int(step_per_epoch * opt.epochs * 0.06)
    opt.total_steps = int(step_per_epoch * opt.epochs)
    logger.info(f"warmup_steps: {opt.warmup_steps}")
    logger.info(f"total_steps: {opt.total_steps}")
    logger.info(f"weight_decay: {opt.weight_decay}")
    optimizer, scheduler = src.util.set_optim(opt, model)
    step, best_dev_em = 0, 0.0

    logger.info("Start training")
    train(
        model,
        optimizer,
        scheduler,
        step,
        train_dataset,
        eval_dataset,
        opt,
        collator,
        best_dev_em,
        checkpoint_path,
        tokenizer
    )
