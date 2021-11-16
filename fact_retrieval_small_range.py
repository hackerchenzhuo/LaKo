# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import csv
import json
import logging
import pickle
import time
import glob
from pathlib import Path

import numpy as np
import torch
import transformers

import src.slurm
import src.util
import src.model
import src.data
import src.index
import time
import pdb
from tqdm import tqdm
import json
import os
import os.path as osp
from src.options import Options
from torch.utils.data import DataLoader

from src.evaluation import calculate_matches

logger = logging.getLogger(__name__)


def embed_questions(opt, data, model, tokenizer):
    batch_size = opt.per_gpu_batch_size
    dataset = src.data.Dataset(data, opt)
    collator = src.data.RetrieverCollator(tokenizer)
    dataloader = DataLoader(dataset, batch_size=batch_size, drop_last=False, num_workers=0, collate_fn=collator, shuffle=False)
    model.eval()
    embedding = []
    with torch.no_grad():
        for k, batch in enumerate(dataloader):
            (idx, question_ids, question_mask, _, _, _) = batch
            output = model.embed_text(
                text_ids=question_ids.view(-1, question_ids.size(-1)).cuda(),
                text_mask=question_mask.view(-1, question_ids.size(-1)).cuda(),
                text_type="q",
                apply_mask=model.config.apply_question_mask,
                extract_cls=model.config.extract_cls,
            )
            embedding.append(output)

    embedding = torch.cat(embedding, dim=0)
    logger.info(f'Questions embeddings shape: {embedding.size()}')

    return embedding.cpu()


def resort_facts(examples, all_id_to_facts_dic, questions_embedding, allembeddings):
    assert len(examples) == questions_embedding.shape[0]
    num = 0
    for ex in tqdm(examples, total=len(examples), desc=f'fact retrieval'):
        questions_embedding_tmp = questions_embedding[num, :]
        num += 1
        fact_ids = [int(i["id"]) for i in ex['fact']]

        fact_ids_torch = torch.LongTensor(fact_ids)

        fact_embedding = torch.index_select(allembeddings, 0, fact_ids_torch)
        score_list = torch.matmul(fact_embedding, questions_embedding_tmp)
        score_list = score_list.numpy().tolist()

        score_list, fact_ids = (list(t) for t in zip(*sorted(zip(score_list, fact_ids), reverse=True)))

        # pdb.set_trace()

        fact_num = len(fact_ids)
        ex['fact'] = [
            {
                'sentence': all_id_to_facts_dic[str(fact_ids[c])],
                'id': fact_ids[c],
                'score': score_list[c]
            } for c in range(fact_num)
        ]


def validate(data, workers_num):
    match_stats = calculate_matches(data, workers_num)
    top_k_hits = match_stats.top_k_hits

    logger.info('Validation results: top k documents hits %s', top_k_hits)
    top_k_hits = [v / len(data) for v in top_k_hits]
    logger.info('Validation results: top k documents hits accuracy %s', top_k_hits)
    return match_stats.questions_doc_hits


def add_facts(data, id_to_fact, top_passages_and_scores):
    # add passages to original data
    merged_data = []
    assert len(data) == len(top_passages_and_scores)
    for i, d in enumerate(data):
        results_and_scores = top_passages_and_scores[i]
        docs = [id_to_fact[doc_id] for doc_id in results_and_scores[0]]
        scores = [int(score) for score in results_and_scores[1]]
        fact_num = len(docs)
        d['fact'] = [
            {
                'sentence': docs[c],
                'id': results_and_scores[0][c],
                'score': scores[c]
            } for c in range(fact_num)
        ]


def add_hasanswer(data, hasanswer):
    # add hasanswer to data
    for i, ex in enumerate(data):
        for k, d in enumerate(ex['ctxs']):
            d['hasanswer'] = hasanswer[i][k]


def main(opt):
    torch.cuda.set_device(opt.gpu)

    src.util.init_logger(is_main=True)
    this_dir = osp.dirname(__file__)
    data_path = osp.abspath(osp.join(this_dir, '..', '..', 'data', 'LaKo'))
    cache_dir = osp.abspath(osp.join(this_dir, '..', '..', 'data', '.cache', 'transformers'))
    tokenizer = transformers.BertTokenizerFast.from_pretrained('bert-base-uncased', cache_dir=cache_dir)
    dir_path = osp.join(data_path, opt.checkpoint_dir)
    write_path = osp.join(dir_path, 'tmp_dir')

    train_data_path = osp.join(data_path, opt.dataset, opt.train_data)
    eval_data_path = osp.join(data_path, opt.dataset, opt.eval_data)
    with open(train_data_path, 'r') as fin:
        train_examples = json.load(fin)
    with open(eval_data_path, 'r') as fin:
        eval_examples = json.load(fin)
    model_class = src.model.Retriever
    opt.model_path = osp.join(data_path, opt.model_path)
    opt.model_path = os.path.realpath(opt.model_path)
    model = model_class.from_pretrained(opt.model_path)
    model.cuda()

    model.eval()

    input_paths = osp.join(write_path, opt.passages_embeddings)
    assert os.path.exists(input_paths)
    with open(input_paths, 'rb') as f:
        allembeddings = pickle.load(f)

    train_questions_embedding = embed_questions(opt, train_examples, model, tokenizer)
    test_questions_embedding = embed_questions(opt, eval_examples, model, tokenizer)

    kg_path_dic = osp.join(data_path, "kg", "v5_id2sentence_dict.json")
    with open(kg_path_dic, 'r') as fin:
        all_id_to_facts_dic = json.load(fin)

    allembeddings = torch.from_numpy(allembeddings)

    resort_facts(train_examples, all_id_to_facts_dic, train_questions_embedding, allembeddings)
    # pdb.set_trace()
    resort_facts(eval_examples, all_id_to_facts_dic, test_questions_embedding, allembeddings)
    # hasanswer = validate(data, args.validation_workers)
    # add_hasanswer(data, hasanswer)

    # pdb.set_trace()

    now_time = time.strftime("%m-%d-%H", time.localtime(time.time()))
    train_data_name = opt.train_data.replace(".json", "")
    test_data_name = opt.eval_data.replace(".json", "")

    output_train_data_path = osp.join(data_path, opt.dataset, f"{train_data_name}_{opt.version}.json")
    output_eval_data_path = osp.join(data_path, opt.dataset, f"{test_data_name}_{opt.version}.json")
    logger.info(f'saving...')
    with open(output_train_data_path, 'w') as fw:
        json.dump(train_examples, fw)
    logger.info(f'Saved results to {output_train_data_path}')
    with open(output_eval_data_path, 'w') as fw:
        json.dump(eval_examples, fw)
    logger.info(f'Saved results to {output_eval_data_path}')


if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    options = Options()
    options.add_retriever_options()
    options.add_optim_options()
    opt = options.parse()
    main(opt)
