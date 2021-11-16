#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import collections
import logging
import regex
import string
import unicodedata
from functools import partial
from multiprocessing import Pool as ProcessPool
from typing import Tuple, List, Dict
import numpy as np
import pdb
"""
Evaluation code from DPR: https://github.com/facebookresearch/DPR
"""
stop_words = ["yes", "no", "which", "and", "this", "we", "what", "the", "can", "are", "likely", "you", "where", "does", 'a', 'he', 'she', 'is', "", "an", "it", "some", "that", "there", 'how', 'other', 'or',
              'bu', 'ha', 'hi', 'wa', 'ga', 'st', 'am', 'cd', 'rv', 'hp', 'uk', 'lo', 'ft', 'dc', 'pm', 'la', 'th', 'vw', 'ly', 'ox', 'my', 'lg', 'dr', '\"i', '\'s', 'mm', 'rd', '3d', 'ny', 'ma', 'aa', 're', 'fo', 'dy', 'nd', 'a ', 'ii', 'ex',
              'av', 'ge', 'dj', 'tp', 'gp', 'os', 'de', 'wi', 'un', 'ct', 'pf', 'ot', 'al', 'co', 'ye', 'hu', 'mt', 'sa', 'bp', 'aw', 'tx', 'ca', 'ne', 'mr', 'jp', 'cb', '\'a', 'fe', 'af', 'ar', 'du', 'od', 'vy', 'fa', 'bi', 'ti', 'si', 'ac', 'pa', 'tw',
              'nw', 'iv', 'lb', '  ', ' ', 'ep', 'op', 'te', '\"e', '\"a', 'hd', 'oj', 'rm', 'a\'', 'o\'', 'ba', 'f5', 'ce', 'yo', 'yo', '#2', 'mn', 'og', 'pt', 'sb', 'ds', '$1', 'em', 'sd', 'ho', 'di', 'pn', 'db', 'ae', '4h', 'cv', 'el', 'rc', 'le', 'v8',
              'kk', 'na', 'vh', 'bt', 'qr', 'om', 'kc', 'ou', 'ln', 'b5', 'pu', 'mo', '\"1', 'ah', 'kg', 'ax', 'pl', 'li', 'sw', 'fc', 'jr', 'sk', 'lf', 'jt', '7,', 'mu', 'aq', 'pj', 'ky', 'jc', 'ab', 'ol', '1.', '2.', 'ay', 'ms', '4,', 'bc', 'bo', 'km', 'ty',
              'll', 'hr', 'oz', 'fi', 'cm', 'yr', 'pb', 'su', 'k9', 'k2', 'sr', 'uv', 'lu', 'j\'', 'mg', 'jk', 'ri', 'md', 'â½', 'hs', 'ed', 'eg', 'fu', 'gb', 'e2', 'sm', 'jo', '\'i', 'fm', 'xl', 'bb', '5g', 'da', 'et', 'ro', 'a1', 'io', 'a2', 's8', 'v1', 'vx',
              'ta', 'ww', 'cy', '4\'', 'h4', 'ie', 'ki', '4e', '#1', 'rt', 'eu', 'ag', 'eo', 'i3', 'o2', 'ea', 'x3', '\'o', 'nn', 'u-', '$2', 'sl', '>>', 'ec', 'nj', 'za', 'ck', 'mc', 'ra', 'ek', '$4', '4o', 'po', 'kw', 'sq', 'mj', 'e\"', 'nu', 'xx', 'b6', 'ei',
              '5%', '1x', 'cn', '\"w', 'm\'', 'i', 'n', 't', 's', 'o', ',', 'm', '"', '&', 'b', 'w', 'e', 'c', 'l', 'y', 'p', '-', 'x', 'd', 'r', 'v', 'g', 'k', 'f', '#', 'h', 'u', 'j', '/', 'q', '!', '@', '(', 'z', ':', '', 'of', 'with']


class SimpleTokenizer(object):
    ALPHA_NUM = r'[\p{L}\p{N}\p{M}]+'
    NON_WS = r'[^\p{Z}\p{C}]'

    def __init__(self):
        """
        Args:
            annotators: None or empty set (only tokenizes).
        """
        self._regexp = regex.compile(
            '(%s)|(%s)' % (self.ALPHA_NUM, self.NON_WS),
            flags=regex.IGNORECASE + regex.UNICODE + regex.MULTILINE
        )

    def tokenize(self, text, uncased=False):
        matches = [m for m in self._regexp.finditer(text)]
        if uncased:
            tokens = [m.group().lower() for m in matches]
        else:
            tokens = [m.group() for m in matches]
        return tokens


logger = logging.getLogger(__name__)

QAMatchStats = collections.namedtuple('QAMatchStats', ['top_k_hits', 'questions_doc_hits'])


def calculate_matches(data: List, workers_num: int):
    """
    Evaluates answers presence in the set of documents. This function is supposed to be used with a large collection of
    documents and results. It internally forks multiple sub-processes for evaluation and then merges results
    :param all_docs: dictionary of the entire documents database. doc_id -> (doc_text, title)
    :param answers: list of answers's list. One list per question
    :param closest_docs: document ids of the top results along with their scores
    :param workers_num: amount of parallel threads to process data
    :param match_type: type of answer matching. Refer to has_answer code for available options
    :return: matching information tuple.
    top_k_hits - a list where the index is the amount of top documents retrieved and the value is the total amount of
    valid matches across an entire dataset.
    questions_doc_hits - more detailed info with answer matches for every question and every retrieved document
    """

    logger.info('Matching answers in top docs...')

    tokenizer = SimpleTokenizer()
    get_score_partial = partial(check_answer, tokenizer=tokenizer)

    processes = ProcessPool(processes=workers_num)
    scores = processes.map(get_score_partial, data)

    logger.info('Per question validation results len=%d', len(scores))

    n_docs = len(data[0]['ctxs'])
    top_k_hits = [0] * n_docs
    for question_hits in scores:
        best_hit = next((i for i, x in enumerate(question_hits) if x), None)
        if best_hit is not None:
            top_k_hits[best_hit:] = [v + 1 for v in top_k_hits[best_hit:]]

    return QAMatchStats(top_k_hits, scores)


def check_answer(example, tokenizer) -> List[bool]:
    """Search through all the top docs to see if they have any of the answers."""
    answers = example['answers']
    ctxs = example['ctxs']

    hits = []

    for i, doc in enumerate(ctxs):
        text = doc['text']

        if text is None:  # cannot find the document for some reason
            logger.warning("no doc in db")
            hits.append(False)
            continue

        hits.append(has_answer(answers, text, tokenizer))

    return hits


def has_answer(answers, text, tokenizer) -> bool:
    """Check if a document contains an answer string."""
    text = _normalize(text)
    text = tokenizer.tokenize(text, uncased=True)

    for answer in answers:
        answer = _normalize(answer)
        answer = tokenizer.tokenize(answer, uncased=True)
        for i in range(0, len(text) - len(answer) + 1):
            if answer == text[i: i + len(answer)]:
                return True
    return False

#################################################
########        READER EVALUATION        ########
#################################################


def _normalize(text):
    return unicodedata.normalize('NFD', text)

# Normalization from SQuAD evaluation script https://worksheets.codalab.org/rest/bundles/0x6b567e1cf2e041ec80d7098f031c5c9e/contents/blob/


def normalize_answer(s, dele_sw=False):

    def remove_stopwords(text):
        ts = text.split()
        for word in ts:
            if word in stop_words:
                text = text.replace(word, "")
        return text

    def remove_articles(text):
        return regex.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    if not dele_sw:
        return white_space_fix(remove_articles(remove_punc(lower(s))))
    else:
        return white_space_fix(remove_stopwords(remove_articles(remove_punc(lower(s)))))


def exact_match_score(prediction, ground_truth, value):
    return (normalize_answer(prediction) == normalize_answer(ground_truth)) * value


def includ_match_score(prediction, ground_truth, value):
    return ((normalize_answer(prediction) in normalize_answer(ground_truth)) or (normalize_answer(ground_truth) in normalize_answer(prediction))) * value


def ems(prediction, ground_truths):
    return max([exact_match_score(prediction, k, v) for k, v in ground_truths.items()])


def includ_ems(prediction, ground_truths):
    return max([includ_match_score(prediction, k, v) for k, v in ground_truths.items()])


def stem_ems(prediction, ground_truths, tokenizer, stemmer, dele_sw=False):
    ans_tokens = tokenizer.tokenize(normalize_answer(prediction, dele_sw))

    stem_ans = set(list(map(stemmer.stem, ans_tokens)))
    score = 0
    ground_truths = sorted(ground_truths.items(), key=lambda x: x[1], reverse=True)
    for ground_truth, value in ground_truths:
        gt_tokens = tokenizer.tokenize(normalize_answer(ground_truth))
        stem_gt = list(map(stemmer.stem, gt_tokens))
        if any(x in stem_ans for x in stem_gt):
            score = value
            break
    return score
####################################################
########        RETRIEVER EVALUATION        ########
####################################################


def eval_batch(scores, inversions, avg_topk, idx_topk):
    for k, s in enumerate(scores):
        s = s.cpu().numpy()
        sorted_idx = np.argsort(-s)
        score(sorted_idx, inversions, avg_topk, idx_topk)


def count_inversions(arr):
    inv_count = 0
    lenarr = len(arr)
    for i in range(lenarr):
        for j in range(i + 1, lenarr):
            if (arr[i] > arr[j]):
                inv_count += 1
    return inv_count


def score(x, inversions, avg_topk, idx_topk):
    x = np.array(x)
    inversions.append(count_inversions(x))
    for k in avg_topk:
        # ratio of passages in the predicted top-k that are
        # also in the topk given by gold score
        avg_pred_topk = (x[:k] < k).mean()
        avg_topk[k].append(avg_pred_topk)
    for k in idx_topk:
        below_k = (x < k)
        # number of passages required to obtain all passages from gold top-k
        idx_gold_topk = len(x) - np.argmax(below_k[::-1])
        idx_topk[k].append(idx_gold_topk)
