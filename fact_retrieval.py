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
    dataloader = DataLoader(dataset, batch_size=batch_size, drop_last=False, num_workers=10, collate_fn=collator, shuffle=False)
    model.eval()
    embedding = []
    with torch.no_grad():
        for k, batch in enumerate(dataloader):
            (idx, question_ids, question_mask, _, _, _) = batch
            output = model.embed_text(
                text_ids=question_ids.view(-1, question_ids.size(-1)).cuda(),
                text_mask=question_mask.view(-1, question_ids.size(-1)).cuda(),
                apply_mask=model.config.apply_question_mask,
                extract_cls=model.config.extract_cls,
            )
            embedding.append(output)

    embedding = torch.cat(embedding, dim=0)
    logger.info(f'Questions embeddings shape: {embedding.size()}')

    return embedding.cpu().numpy()


def index_encoded_data(index, embedding_files, indexing_batch_size):
    with open(embedding_files, 'rb') as f:
        allembeddings = pickle.load(f)
    logger.info(f'Loading file {embedding_files} done!')
    allids = list(range(allembeddings.shape[0]))

    while allembeddings.shape[0] > indexing_batch_size:
        allembeddings, allids = add_embeddings(index, allembeddings, allids, indexing_batch_size)

    while allembeddings.shape[0] > 0:
        allembeddings, allids = add_embeddings(index, allembeddings, allids, indexing_batch_size)

    logger.info('Data indexing completed.')


def add_embeddings(index, embeddings, ids, indexing_batch_size):
    assert embeddings.shape[0] == len(ids)
    end_idx = min(indexing_batch_size, embeddings.shape[0])
    ids_toadd = ids[:end_idx]
    embeddings_toadd = embeddings[:end_idx]
    ids = ids[end_idx:]
    embeddings = embeddings[end_idx:]
    index.index_data(ids_toadd, embeddings_toadd)
    return embeddings, ids


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

    index = src.index.Indexer(model.config.indexing_dimension)
    # index = src.index.Indexer(model.config.indexing_dimension, opt.n_subquantizers, opt.n_bits)
    input_paths = osp.join(write_path, opt.passages_embeddings)
    assert os.path.exists(input_paths)

    # input_paths = sorted(input_paths)

    embeddings_dir = osp.join(write_path, 'faiss_dir')
    if not os.path.exists(embeddings_dir):
        os.mkdir(embeddings_dir)

    index_path = osp.join(embeddings_dir, 'index.faiss')

    if opt.save_or_load_index and os.path.exists(index_path):
        src.index.deserialize_from(embeddings_dir)
    else:
        logger.info(f'Indexing passages from files {input_paths}')
        start_time_indexing = time.time()
        index_encoded_data(index, input_paths, opt.indexing_batch_size)
        logger.info(f'Indexing time: {time.time()-start_time_indexing:.1f} s.')
        if opt.save_or_load_index:
            src.index.serialize(embeddings_dir)

    train_questions_embedding = embed_questions(opt, train_examples, model, tokenizer)
    test_questions_embedding = embed_questions(opt, eval_examples, model, tokenizer)

    # get top k results
    n_docs = 500
    start_time_retrieval = time.time()
    train_top_ids_and_scores = index.search_knn(train_questions_embedding, n_docs)
    logger.info(f'train Search time: {time.time()-start_time_retrieval:.1f} s.')

    start_time_retrieval = time.time()
    test_top_ids_and_scores = index.search_knn(test_questions_embedding, n_docs)
    logger.info(f'test Search time: {time.time()-start_time_retrieval:.1f} s.')

    kg_path = osp.join(data_path, "kg", "v5_id2sentence.json")
    kg_path_dic = osp.join(data_path, "kg", "v5_id2sentence_dict.json")
    with open(kg_path, 'r') as fin:
        all_id_to_facts = json.load(fin)
    with open(kg_path_dic, 'r') as fin:
        all_id_to_facts_dic = json.load(fin)

    add_facts(train_examples, all_id_to_facts_dic, train_top_ids_and_scores)
    # pdb.set_trace()
    add_facts(eval_examples, all_id_to_facts_dic, test_top_ids_and_scores)
    # hasanswer = validate(data, args.validation_workers)
    # add_hasanswer(data, hasanswer)

    # pdb.set_trace()
    now_time = time.strftime("%m-%d-%H", time.localtime(time.time()))
    train_data_name = opt.train_data.replace(".json", "")
    test_data_name = opt.eval_data.replace(".json", "")

    output_train_data_path = osp.join(data_path, opt.dataset, f"{train_data_name}_{now_time}.json")
    output_eval_data_path = osp.join(data_path, opt.dataset, f"{test_data_name}_{now_time}.json")

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

    # parser.add_argument('--data', required=True, type=str, default=None,
    #                     help=".json file containing question and answers, similar format to reader data")
    # parser.add_argument('--passages', type=str, default=None, help='Path to passages (.tsv file)')
    # parser.add_argument('--passages_embeddings', type=str, default='fact_embedding_dim256_at_11-07-13.pkl', help='Glob path to encoded passages')
    # parser.add_argument('--output_path', type=str, default=None, help='Results are written to output_path')
    # parser.add_argument('--n-docs', type=int, default=100, help="Number of documents to retrieve per questions")
    # parser.add_argument('--validation_workers', type=int, default=32,
    #                     help="Number of parallel processes to validate results")
    # parser.add_argument('--per_gpu_batch_size', type=int, default=64, help="Batch size for question encoding")

    # parser.add_argument('--model_path', type=str, help="path to directory containing model weights and config file")
    # parser.add_argument('--no_fp16', action='store_true', help="inference in fp32")
    # parser.add_argument('--passage_maxlength', type=int, default=200, help="Maximum number of tokens in a passage")
    # parser.add_argument('--question_maxlength', type=int, default=40, help="Maximum number of tokens in a question")

    # args = parser.parse_args()
    opt = options.parse()
    # src.slurm.init_distributed_mode(args)
    main(opt)
