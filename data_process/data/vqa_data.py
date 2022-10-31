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

# Load part of the dataset for fast checking.
# Notice that here is the number of images instead of the number of data,
# which means all related data to the images would be used.
TINY_IMG_NUM = 512
FAST_IMG_NUM = 5000

# The path to data and image features.
# VQA_DATA_ROOT = 'data/vqa/'
#

SPLIT2NAME = {
    'train': 'train2014',
    'valid': 'val2014',
    'minival': 'val2014',
    'nominival': 'val2014',
    'test': 'test2015',
}


class VQADataset:
    """
    A VQA data example in json file:
        {
            "answer_type": "other",
            "img_id": "COCO_train2014_000000458752",
            "fact": ["cat","eat","fish"]
            "label": {
                "net": 1
            },
            "question_id": 458752000,
            "question_type": "what is this",
            "sent": "What is this photo taken looking through?"
        }
    """
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
        self.id2datum = {}
        for datum in tqdm(self.data, desc="deal data.."):
            datum['fact'] = ""
            # Convert list to dict (for evaluation)
            self.id2datum[datum['question_id']] = datum

        logger.info("Load %d data from split(s) %s." % (len(self.data), self.name))

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
        return data[item]


"""
An example in obj36 tsv:
FIELDNAMES = ["img_id", "img_h", "img_w", "objects_id", "objects_conf",
              "attrs_id", "attrs_conf", "num_boxes", "boxes", "features"]
FIELDNAMES would be keys in the dict returned by load_obj_tsv.
"""


class VQATorchDataset(Dataset):
    def __init__(self, args, logger, dataset):
        super().__init__()
        self.raw_dataset = dataset

        if args.tiny:
            logger.info("-- tiny valid data")
            topk = TINY_IMG_NUM
        elif args.fast:
            logger.info("-- fast valid data")
            topk = FAST_IMG_NUM
        else:
            logger.info("-- full valid data")
            topk = None  # 默认

        self.img_list = None

        if args.dataset != 'vqa2.0':
            # cache img id
            self.cache_img(args, logger)

        # Loading detection features to img_data
        self.img_data = []

        MSCOCO_IMGFEAT_ROOT = osp.join(args.data_root, 'common_data/images/lxmert_mscoco_imgfeat')  # 包含 '%s_obj36.tsv'的文件夹

        for split in dataset.splits:
            # Minival is 5K images in MS COCO, which is used in evaluating VQA/LXMERT-pre-training.
            # It is saved as the top 5K features in val2014_***.tsv
            load_topk = 5000 if (split == 'minival' and topk is None) else topk

            self.img_data.extend(load_obj_tsv(
                os.path.join(MSCOCO_IMGFEAT_ROOT, '%s_obj36.tsv' % (SPLIT2NAME[split])),
                topk=load_topk, name=SPLIT2NAME[split], img_list=self.img_list, args=args))

        # Convert img list to dict
        self.imgid2img = {}
        for img_datum in self.img_data:
            # 处理过： "img_id" : COCO_val2014_000000338207 -> 338207
            # if args.vqa_test and args.dataset == 'vqa2.0':

            #     img_datum['img_id'] = int(img_datum["img_id"].split('_')[-1])
            self.imgid2img[img_datum['img_id']] = img_datum

        # Only kept the data with loaded image features
        self.data = []

        for datum in self.raw_dataset.data:
            if args.vqa_test and args.dataset == 'vqa2.0' and args.test == "test":
                datum['img_id'] = int(datum['img_id'].split('_')[-1])

            if datum['img_id'] in self.imgid2img:
                self.data.append(datum)

        logger.info("Use %d data in torch dataset" % (len(self.data)))

    def cache_img(self, args, logger):
        self.img_list = []
        file_name = args.dataset + "_"
        for split in self.raw_dataset.splits:
            file_name = file_name + split + "_"
        file_name = file_name + "img_cand.json"

        file_path = osp.join(args.dataset_root, "cache", file_name)

        if not osp.exists(file_path):
            logger.info(f"generate {file_name} cache file...")
            for datum in self.raw_dataset.data:
                # 需要的image id 存下来
                self.img_list.append(datum['img_id'])
            with open(file_path, "w") as fp:
                json.dump(self.img_list, fp)
        else:
            logger.info(f"loading {file_name} obj_36 cache file...")
            with open(file_path, "r") as fp:
                self.img_list = json.load(fp)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item: int):
        datum = self.data[item]

        img_id = datum['img_id']
        ques_id = datum['question_id']
        ques = datum['sent']
        # Get image info
        img_info = self.imgid2img[img_id]
        obj_num = img_info['num_boxes']
        feats = img_info['features'].copy()
        boxes = img_info['boxes'].copy()
        assert obj_num == len(boxes) == len(feats)

        # Normalize the boxes (to 0 ~ 1)
        img_h, img_w = img_info['img_h'], img_info['img_w']
        boxes = boxes.copy()
        boxes[:, (0, 2)] /= img_w
        boxes[:, (1, 3)] /= img_h
        np.testing.assert_array_less(boxes, 1 + 1e-5)
        np.testing.assert_array_less(-boxes, 0 + 1e-5)

        # Provide label (target)
        if 'label' in datum:
            label = datum['label']
            target = torch.zeros(self.raw_dataset.num_answers)
            for ans, score in label.items():
                target[self.raw_dataset.ans2label[ans]] = score
            if 'fact' in datum:
                fact = datum['fact']
                return ques_id, feats, boxes, ques, fact, target
            else:
                return ques_id, feats, boxes, ques, target
        else:
            if 'fact' in datum:
                fact = datum['fact']
                return ques_id, feats, boxes, ques, fact
            else:
                return ques_id, feats, boxes, ques

    def updata_question(self, dataset):
        for datum in self.data:
            datum['sent'] = dataset.id2datum[datum['question_id']]['sent']


class VQAEvaluator:
    def __init__(self, args, logger, dataset):
        self.dataset = dataset
        self.args = args

    def evaluate(self, quesid2ans: dict):
        score = 0.
        for quesid, ans in quesid2ans.items():
            datum = self.dataset.id2datum[quesid]
            label = datum['label']
            if ans in label:
                score += label[ans]
        return score / len(quesid2ans)

    def dump_result(self, quesid2ans: dict, path):
        """
        Dump results to a json file, which could be submitted to the VQA online evaluation.
        VQA json file submission requirement:
            results = [result]
            result = {
                "question_id": int,
                "answer": str
            }

        :param quesid2ans: dict of quesid --> ans
        :param path: The desired path of saved file.
        """
        # 详细可读的结果
        if self.args.detail_dump_result:
            with open(path, 'w') as f:
                result = []
                for ques_id, ans in quesid2ans.items():
                    datum = self.dataset.id2datum[ques_id]
                    ques = datum['sent']
                    img_id = datum['img_id']
                    result.append({
                        'question_id': ques_id,
                        'question': ques,
                        'img_id': img_id,
                        'answer': ans
                    })

                json.dump(result, f, indent=4, sort_keys=True)
        else:
            with open(path, 'w') as f:
                result = []
                for ques_id, ans in quesid2ans.items():
                    result.append({
                        'question_id': ques_id,
                        'answer': ans
                    })
                json.dump(result, f, indent=4, sort_keys=True)
