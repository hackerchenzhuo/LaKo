# coding=utf-8
# Copyleft 2019 project LXRT.

import json
import os
import pickle
from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import Dataset
import os.path as osp
from .utils import load_obj_tsv
import numpy as np
import pdb
import random
from itertools import groupby
from transformers import LxmertTokenizerFast
# Load part of the dataset for fast checking.
# Notice that here is the number of images instead of the number of data,
# which means all related data to the images would be used.
TINY_IMG_NUM = 512
FAST_IMG_NUM = 5000

# The path to data and image features.
# VQA_DATA_ROOT = 'data/vqa/'
#
MAX_CONTEXT_LEN = 50
CLS = "[CLS]"
SEP = "[SEP]"

SPLIT2NAME = {
    'train': 'train2014',
    'valid': 'val2014',
    'minival': 'val2014',
    'nominival': 'val2014',
    'test': 'test2015',
}


class VQAPromptDataset:
    # note:  "img_id": "458752", here
    def __init__(self, args, logger, splits: str):
        self.name = splits
        self.splits = splits.split(',')
        self.cache_path = osp.join(args.dataset_root, 'cache', str(int(args.min_occurence)))  # /data/chenzhuo/data/KPVQA/vqa2.0/cache
        # Loading datasets
        self.data = []
        for split in self.splits:
            self.data.extend(json.load(open(osp.join(self.cache_path, f'{split}.json'), 'r')))
            # self.data.extend(json.load(open("data/vqa/%s.json" % split)))
        logger.info("Load %d data from split(s) %s." % (len(self.data), self.name))
        logger.info("Prompt setting!")

        self.sep = SEP

        # Answers

        self.ans2label = json.load(open(osp.join(self.cache_path, 'trainval_ans2label.json'), 'r'))
        self.label2ans = json.load(open(osp.join(self.cache_path, 'trainval_label2ans.json'), 'r'))
        assert len(self.ans2label) == len(self.label2ans)
        # # BUG FIXED: 新版本pythpn不支持对于list的跨度索引
        # self.label2ans = np.array(self.label2ans)

    @property
    def num_answers(self):
        return len(self.ans2label)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item: int):
        return self.data[item]


class VQAPrompt_1_Dataset(VQAPromptDataset):
    """
        Fact: {best ans}. [SEP] Question: [x]
    """

    def __init__(self, args, logger, splits: str):
        super().__init__(args, logger, splits)
        # Convert list to dict (for evaluation)
        # cache_dir = '/data/chenzhuo/data/.cache/transformers'
        # tokenizer = LxmertTokenizerFast.from_pretrained("unc-nlp/lxmert-base-uncased", cache_dir = cache_dir)
        # ques_len=[]

        self.id2datum = {}
        for datum in tqdm(self.data, desc="Add best ans prompt"):
            if 'label' in datum:
                label = datum['label']
                ans = ""
                score = 0
                for _ans, _score in label.items():
                    if _score > score:
                        ans = _ans
                        score = _score

                # prompt
                if not args.split_segment:
                    datum['sent'] = f"Fact: {ans}. {self.sep} Question: {datum['sent']}"
                    datum['fact'] = ""
                else:
                    datum['sent'] = f"Question: {datum['sent']}"
                    datum['fact'] = f"Fact: {ans}."

            self.id2datum[datum['question_id']] = datum
            # ques_len.append(tokenizer(datum['sent'], return_tensors='pt')["input_ids"].shape[1])
            # pdb.set_trace()

        # ques_len_all =len(ques_len)
        # for k,g in groupby(sorted(ques_len),key=lambda x:x//5):
        #     print(f'{k*5}-{(k+1)*5-1}:{len(list(g))/ques_len_all * 100: .4}%')
        # pdb.set_trace()
        # prompt question -1
            # len----vqa2.0------ 30
            # 10-14: 35.62%
            # 15-19: 59.36%
            # 20-24: 4.69%
            # 25-29: 0.3108%
            # 30-34: 0.01758%
            # len----okvqa 10------ 32
            # 10-14: 16.54%
            # 15-19: 64.01%
            # 20-24: 16.54%
            # 25-29: 2.542%
            # 30-34: 0.3108%
            # 35-39: 0.0444%
            # 40-44: 0.0111%
            # len----okvqa 3------ 32
            # 10-14: 12.41%
            # 15-19: 65.22%
            # 20-24: 18.95%
            # 25-29: 2.953%
            # 30-34: 0.3774%
            # 35-39: 0.0777%
            # 40-44: 0.0111%

        # normal question
            # len----vqa2.0------ 20
            # 5-9: 58.03%
            # 10-14: 39.48%
            # 15-19: 2.3%
            # 20-24: 0.1755%
            # 25-29: 0.006535%
            # len----okvqa------ 24
            # 5-9: 31.96%
            # 10-14: 53.87%
            # 15-19: 12.04%
            # 20-24: 1.832%
            # 25-29: 0.2664%
            # 30-34: 0.0222%
            # 35-39: 0.0111%


class VQAPrompt_2_Dataset(VQAPromptDataset):
    """
        Fact: {all ans ( from big to small )}. [SEP] Question: [x]
    """
    # note:  "img_id": "458752", here

    def __init__(self, args, logger, splits: str):
        super().__init__(args, logger, splits)

        self.id2datum = {}

        for datum in tqdm(self.data, desc="Add all ans prompt"):
            if 'label' in datum:
                label = datum['label']
                label_order = sorted(label.items(), key=lambda x: x[1], reverse=True)
                ans = ""
                label_len = len(label_order)
                for i in range(label_len):
                    _ans, _ = label_order[i]
                    ans += _ans
                    if i < label_len - 1:
                        ans += ", "

                # prompt
                if not args.split_segment:
                    datum['sent'] = f"Fact: {ans}. {self.sep} Question: {datum['sent']}"
                    datum['fact'] = ""
                else:
                    datum['sent'] = f"Question: {datum['sent']}"
                    datum['fact'] = f"Fact: {ans}."

            self.id2datum[datum['question_id']] = datum

        # prompt question -2
            # len----vqa2.0------ 34
            # 10-19: 71.04%
            # 20-29: 27.35%
            # 30-39: 1.59%
            # 40-49: 0.02096%
            # len----okvqa 10----- 34
            # 10-14: 11.72%
            # 15-19: 57.68%
            # 20-24: 25.43%
            # 25-29: 4.507%
            # 30-34: 0.5883%
            # 35-39: 0.0666%
            # 40-44: 0.0111%
            # len----okvqa 3----- 34
            # 10-14: 4.962%
            # 15-19: 44.86%
            # 20-24: 38.55%
            # 25-29: 10.08%
            # 30-34: 1.354%
            # 35-39: 0.1887%
            # 40-44: 0.0111%


class VQAPrompt_3_Dataset(VQAPromptDataset):
    """
        Fact: {rand ans}. [SEP] Question: [x]
    """
    # note:  "img_id": "458752", here

    def __init__(self, args, logger, splits: str):
        super().__init__(args, logger, splits)

        self.id2datum = {}

        for datum in tqdm(self.data, desc="Add rand ans prompt"):
            if 'label' in datum:
                label = datum['label']
                key_list = list(label.keys())
                ans = ""
                if len(key_list) > 0:
                    ans = random.choice(key_list)

                # prompt
                if not args.split_segment:
                    datum['sent'] = f"Fact: {ans}. {self.sep} Question: {datum['sent']}"
                    datum['fact'] = ""
                else:
                    datum['sent'] = f"Question: {datum['sent']}"
                    datum['fact'] = f"Fact: {ans}."

            self.id2datum[datum['question_id']] = datum
