# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import json
import logging
from collections import defaultdict
import numpy as np
import torch
import nltk.tokenize as tk
import nltk.stem.porter as pt
import src.util
from tqdm import tqdm
from src.options import Options
import json
import os.path as osp
import pdb
from numpy import mean

# from src.evaluation import calculate_matches
import src.evaluation

logger = logging.getLogger(__name__)


def calculate_matches(data):
    tk_tokenizer = tk.WordPunctTokenizer()
    pt_stemmer = pt.PorterStemmer()

    hitk = [5, 10, 20, 50, 100, 150, 200, 300, 400, 500]
    max_k = max(hitk)
    score_dict = defaultdict(list)
    score_dict_stem = defaultdict(list)
    len_data = len(data)
    # pdb.set_trace()
    for example in tqdm(data, total=len_data, desc=f'Eval_retri_fact'):
        k = 0
        includ_score = 0
        stem_score = 0
        gold = example['answer']
        facts = example['fact']
        for fact in facts:
            k += 1
            if k > max_k:
                break
            if includ_score < 1:
                includ_score = max(src.evaluation.includ_ems(fact['sentence'], gold), includ_score)
            if stem_score < 1:
                stem_score = max(src.evaluation.stem_ems(fact['sentence'], gold, tk_tokenizer, pt_stemmer, dele_sw=True), stem_score)
            elif includ_score == 1 and stem_score == 1:
                for j in hitk:
                    if j >= k:
                        score_dict[j].append(1)
                        score_dict_stem[j].append(1)
                k = max_k
                break

            if k in hitk:
                score_dict[k].append(includ_score)
                score_dict_stem[k].append(stem_score)
        if k < max_k:
            for j in hitk:
                if j > k:
                    score_dict[j].append(includ_score)
                    score_dict_stem[j].append(stem_score)

    score_output = {}
    score_output_stem = {}
    for k in hitk:
        assert len(score_dict[k]) == len_data and len(score_dict_stem[k]) == len_data
        score_output[k] = mean(score_dict[k])
        score_output_stem[k] = mean(score_dict_stem[k])
    return score_output, score_output_stem


def validate(data):
    # top-k retrieval accuracy (P@k), which is the percentage of questions for which
    # at least one passage of the top-k retrieved passages contains the gold answer.
    # It is unclear how well this metric evaluates the retriever performance,
    # since the answer can be contained in a passage without being related to the question.
    # This is notably true for common words or entities

    score_hit_n, score_hit_n_stem = calculate_matches(data)

    for key in score_hit_n.keys():
        logger.info(f'Validation results: top {key} facts hits {100*score_hit_n[key]:.2f}')

    for key in score_hit_n_stem.keys():
        logger.info(f'Validation results: top {key} facts hits {100*score_hit_n_stem[key]:.2f} (stem based)')

    # top_k_hits = [v / len(data) for v in top_k_hits]
    # logger.info('Validation results: top k documents hits accuracy %s', top_k_hits)
    # return match_stats.questions_doc_hits


def main(opt):
    logger = src.util.init_logger(is_main=True)
    this_dir = osp.dirname(__file__)
    data_path = osp.abspath(osp.join(this_dir, '..', '..', 'data', 'LaKo'))
    cache_dir = osp.abspath(osp.join(this_dir, '..', '..', 'data', '.cache', 'transformers'))

    eval_data = opt.eval_data.replace(".json", "")
    eval_data_path = osp.join(data_path, opt.dataset, f"{eval_data}_{opt.version}.json")
    if not osp.exists(eval_data_path):
        logger.info(f"path:{eval_data_path} not exist!")
        eval_data_path = osp.join(data_path, opt.dataset, opt.eval_data)
    assert osp.exists(eval_data_path)
    with open(eval_data_path, 'r') as fin:
        eval_examples = json.load(fin)

    # answers = [ex['answer'] for ex in eval_examples]
    questions_doc_hits = validate(eval_examples)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    options = Options()

    options.add_optim_options()
    options.add_retriever_options()
    opt = options.parse()
    # torch.cuda.set_device(opt.gpu)
    main(opt)
