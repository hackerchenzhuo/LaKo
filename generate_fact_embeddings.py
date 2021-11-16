# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import argparse
import csv
import logging
import pickle
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from src.options import Options
import transformers

import src.model
import src.data
import src.util
import src.slurm
import time
import os.path as osp
import pdb
from tqdm import tqdm
import json

logger = logging.getLogger(__name__)


def embed_passages(opt, all_id_to_facts, model, tokenizer):
    allembeddings = torch.zeros(300600, model.config.indexing_dimension)
    collator = src.data.TextCollator(tokenizer)
    dataset = src.data.TextDataset(all_id_to_facts)
    dataloader = DataLoader(dataset, batch_size=opt.per_gpu_batch_size, drop_last=False, num_workers=12, collate_fn=collator, shuffle=False)
    total = 0
    allids = []
    # len_allids_set = 0
    with torch.no_grad():
        for k, (ids, text_ids, text_mask) in tqdm(enumerate(dataloader), total=len(dataloader), desc=f'generating'):

            embeddings = model.embed_text(
                text_ids=text_ids.cuda(),
                text_mask=text_mask.cuda(),
                text_type="f",
                apply_mask=model.config.apply_passage_mask,
                extract_cls=model.config.extract_cls,
            )
            embeddings = embeddings.cpu()

            sz = len(ids)
            for i in range(sz):
                allembeddings[ids[i]][:] = embeddings[i][:]

            allids.extend(ids)
            # len_allids_set = len(allids_set)
            # if k >= 5:
            #     break
    return allembeddings.numpy()


def main(opt):
    logger = src.util.init_logger(is_main=True)
    this_dir = osp.dirname(__file__)
    data_path = osp.abspath(osp.join(this_dir, '..', '..', 'data', 'LaKo'))
    cache_dir = osp.abspath(osp.join(this_dir, '..', '..', 'data', '.cache', 'transformers'))

    tokenizer = transformers.BertTokenizerFast.from_pretrained('bert-base-uncased', cache_dir=cache_dir)
    # config = src.model.RetrieverConfig(
    #     indexing_dimension=opt.indexing_dimension,
    #     apply_question_mask=not opt.no_question_mask,
    #     apply_passage_mask=not opt.no_passage_mask,
    #     extract_cls=opt.extract_cls,
    #     projection=not opt.no_projection,
    # )

    # model_class = src.model.Retriever(config, cache_dir=cache_dir)
    model_class = src.model.Retriever
    #

    # passages = src.util.load_passages(args.passages)
    dir_path = osp.join(data_path, opt.checkpoint_dir)
    write_path = osp.join(dir_path, 'tmp_dir')

    train_data_path = osp.join(data_path, opt.dataset, opt.train_data)
    eval_data_path = osp.join(data_path, opt.dataset, opt.eval_data)

    # with open(train_data_path, 'r') as fin:
    #     train_examples = json.load(fin)
    # with open(eval_data_path, 'r') as fin:
    #     eval_examples = json.load(fin)
    # train_examples.extend(eval_examples)
    # del eval_examples
    kg_path = osp.join(data_path, "kg", "v5_id2sentence.json")
    assert os.path.exists(kg_path)
    with open(kg_path, 'r') as fin:
        all_id_to_facts = json.load(fin)
    end_idx = 300600

    opt.model_path = osp.join(data_path, opt.model_path)
    opt.model_path = os.path.realpath(opt.model_path)

    model = model_class.from_pretrained(opt.model_path)
    model.eval()
    model = model.cuda()
    # model = None

    allembeddings = embed_passages(opt, all_id_to_facts, model, tokenizer)

    now_time = time.strftime("%m-%d-%H", time.localtime(time.time()))
    # pdb.set_trace()
    output_name = f"fact_embedding_dim{model.config.indexing_dimension}_at_{opt.version}.pkl"
    output_path = osp.join(write_path, output_name)

    # output_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f'Saving embeddings to {output_path}')

    # pdb.set_trace()

    with open(output_path, mode='wb') as f:
        pickle.dump(allembeddings, f)

    logger.info(f'Saving done.')


if __name__ == '__main__':

    options = Options()
    # options.add_reader_options()
    options.add_optim_options()
    options.add_retriever_options()
    opt = options.parse()
    torch.cuda.set_device(opt.gpu)
    main(opt)
