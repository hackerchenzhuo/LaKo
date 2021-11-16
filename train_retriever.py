# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import time
import sys
import torch
import transformers
from pathlib import Path
import numpy as np
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler, SequentialSampler

import os
import os.path as osp
import copy
import json
import pdb

from tqdm import tqdm
import nltk.tokenize as tk
import nltk.stem.porter as pt
import src.slurm
import src.util
import src.evaluation
import src.data
import src.model
from src.options import Options

import warnings
warnings.filterwarnings("ignore")


def train(model, optimizer, scheduler, global_step, train_dataset, dev_dataset, opt, collator, best_eval_loss):

    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        batch_size=opt.per_gpu_batch_size,
        drop_last=True,
        num_workers=12,
        collate_fn=collator
    )

    loss, curr_loss = 0.0, 0.0
    epoch = 0
    step_per_epoch = int(len(train_dataloader) / opt.per_gpu_batch_size)
    model.train()
    patience = 0

    tk_tokenizer = tk.WordPunctTokenizer()
    pt_stemmer = pt.PorterStemmer()

    while epoch < opt.epochs:
        epoch += 1
        for i, batch in tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc=f'Training | epoch {epoch}'):
            global_step += 1
            (idx, question_ids, question_mask, passage_ids, passage_mask, gold_score) = batch
            _, _, _, train_loss = model(
                question_ids=question_ids.cuda(),
                question_mask=question_mask.cuda(),
                passage_ids=passage_ids.cuda(),
                passage_mask=passage_mask.cuda(),
                gold_score=gold_score.cuda(),
            )

            train_loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), opt.clip)
            optimizer.step()
            scheduler.step()
            model.zero_grad()

            train_loss = src.util.average_main(train_loss, opt)
            curr_loss += train_loss.item()

            del train_loss

        patience += 1
        eval_loss, inversions, avg_topk, idx_topk = evaluate(model, dev_dataset, collator, opt)
        if opt.is_main:
            log = f"{global_step} / {opt.total_steps}"
            log += f" -- train: {curr_loss/step_per_epoch:.6f}"
            log += f", eval: {eval_loss:.6f}"
            log += f", inv: {inversions:.1f}"
            log += f", lr: {scheduler.get_last_lr()[0]:.6f}"
            for k in avg_topk:
                log += f" | avg top{k}: {100*avg_topk[k]:.1f}"
            for k in idx_topk:
                log += f" | idx top{k}: {idx_topk[k]:.1f}"
            logger.info(log)

            curr_loss = 0

        if eval_loss < best_eval_loss:
            patience = 0
            best_eval_loss = eval_loss
            if opt.is_main:
                src.util.save(model, optimizer, scheduler, global_step, best_eval_loss, opt, dir_path, 'best_dev')

        if patience > opt.early_stop:
            logger.info(f"early stop in epoch {epoch}")
            break
        model.train()

    log = f"stop epoch {epoch} |"
    log += f"best_eval_loss: {best_eval_loss:.4f}EM |"


def evaluate(model, dataset, collator, opt):
    sampler = SequentialSampler(dataset)
    dataloader = DataLoader(
        dataset,
        sampler=sampler,
        batch_size=opt.per_gpu_batch_size,
        drop_last=False,
        num_workers=10,
        collate_fn=collator
    )
    model.eval()
    if hasattr(model, "module"):
        model = model.module
    total = 0
    eval_loss = []

    avg_topk = {k: [] for k in [1, 2, 5] if k <= opt.n_context}
    idx_topk = {k: [] for k in [1, 2, 5] if k <= opt.n_context}

    inversions = []
    with torch.no_grad():
        for i, batch in tqdm(enumerate(dataloader), total=len(dataloader), desc=f'Eval'):
            (idx, question_ids, question_mask, context_ids, context_mask, gold_score) = batch

            _, _, scores, loss = model(
                question_ids=question_ids.cuda(),
                question_mask=question_mask.cuda(),
                passage_ids=context_ids.cuda(),
                passage_mask=context_mask.cuda(),
                gold_score=gold_score.cuda(),
            )

            src.evaluation.eval_batch(scores, inversions, avg_topk, idx_topk)
            total += question_ids.size(0)

    inversions = src.util.weighted_average(np.mean(inversions), total, opt)[0]
    for k in avg_topk:
        avg_topk[k] = src.util.weighted_average(np.mean(avg_topk[k]), total, opt)[0]
        idx_topk[k] = src.util.weighted_average(np.mean(idx_topk[k]), total, opt)[0]

    return loss, inversions, avg_topk, idx_topk


if __name__ == "__main__":
    options = Options()
    options.add_retriever_options()
    options.add_optim_options()
    opt = options.parse()
    torch.manual_seed(opt.seed)
    src.slurm.init_distributed_mode(opt)
    src.slurm.init_signal_handler()

    this_dir = osp.dirname(__file__)
    data_path = osp.abspath(osp.join(this_dir, '..', '..', 'data', 'LaKo'))
    cache_dir = osp.abspath(osp.join(this_dir, '..', '..', 'data', '.cache', 'transformers'))
    torch.cuda.set_device(opt.gpu)
    opt.device = opt.gpu
    if opt.model_path == "none":
        from_scratch = "_from_scratch"
    else:
        from_scratch = ""
    if opt.use_fact == "yes":
        fact_para = f"_content_{opt.n_context}"
    else:
        fact_para = ""

    model_name = f"retriever_{opt.dataset}_batch_{opt.per_gpu_batch_size}{fact_para}{from_scratch}_{opt.version}"
    opt.name = model_name

    dir_path = osp.join(data_path, opt.checkpoint_dir, opt.name)
    os.makedirs(dir_path, exist_ok=True)

    log_dir = osp.join(dir_path, 'run.log')

    logger = src.util.init_logger(opt.is_main, opt.is_distributed, log_dir)

    # Load data
    tokenizer = transformers.BertTokenizerFast.from_pretrained('bert-base-uncased', cache_dir=cache_dir)
    collator_function = src.data.RetrieverCollator(
        tokenizer,
        passage_maxlength=opt.passage_maxlength,
        question_maxlength=opt.question_maxlength
    )

    train_data_path = osp.join(data_path, opt.checkpoint_dir, "tmp_dir", opt.train_data)
    eval_data_path = osp.join(data_path, opt.checkpoint_dir, "tmp_dir", opt.eval_data)
    with open(train_data_path, 'r') as fin:
        train_examples = json.load(fin)
    train_dataset = src.data.Dataset(train_examples, opt)
    with open(eval_data_path, 'r') as fin:
        eval_examples = json.load(fin)
    eval_dataset = src.data.Dataset(eval_examples, opt)

    global_step = 0
    best_eval_loss = np.inf
    if opt.asymmetric_retri == "yes":
        opt.no_projection = True
        logger.info(f"using asymmetric retriever...")

    config = src.model.RetrieverConfig(
        indexing_dimension=opt.indexing_dimension,
        apply_question_mask=not opt.no_question_mask,
        apply_passage_mask=not opt.no_passage_mask,
        extract_cls=opt.extract_cls,
        projection=not opt.no_projection,
        asymmetric_retri=opt.asymmetric_retri,
    )
    model_class = src.model.Retriever
    if opt.model_path == "none":
        model = model_class(config, initialize_wBERT=True)
        src.util.set_dropout(model, opt.dropout)
    else:
        opt.model_path = osp.join(data_path, opt.model_path)
        model, optimizer, scheduler, opt_checkpoint, global_step, best_eval_loss = src.util.load(model_class, opt.model_path, opt, reset_params=True)
        logger.info(f"Model loaded from {opt.model_path}")

    model = model.cuda()

    step_per_epoch = int(len(train_dataset) / opt.per_gpu_batch_size)
    opt.warmup_steps = int(step_per_epoch * opt.epochs * 0.06)
    opt.total_steps = int(step_per_epoch * opt.epochs)
    logger.info(f"warmup_steps: {opt.warmup_steps}")
    logger.info(f"total_steps: {opt.total_steps}")
    logger.info(f"weight_decay: {opt.weight_decay}")

    optimizer, scheduler = src.util.set_optim(opt, model)

    train(
        model,
        optimizer,
        scheduler,
        global_step,
        train_dataset,
        eval_dataset,
        opt,
        collator_function,
        best_eval_loss
    )
