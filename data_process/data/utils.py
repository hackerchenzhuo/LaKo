# coding=utf-8
# Copyleft 2019 Project LXRT
import os
import os.path as osp
import sys
import csv
import base64
import time
from tqdm import tqdm
import numpy as np
import json
import pickle
import pdb

csv.field_size_limit(sys.maxsize)
FIELDNAMES = ["img_id", "img_h", "img_w", "objects_id", "objects_conf",
              "attrs_id", "attrs_conf", "num_boxes", "boxes", "features"]


def load_obj_tsv(fname, topk=None, name = "", img_list=None, args=None ):
    """Load object features from tsv file.

    :param fname: The path to the tsv file.
    :param topk: Only load features for top K images (lines) in the tsv file.
        Will load all the features if topk is either -1 or None.
    :return: A list of image object features where each feature is a dict.
        See FILENAMES above for the keys in the feature dict.
    """
    tiny_name = ""
    fast_name = ""
    assert not (args.fast and args.tiny)
    if name != "train2014":
        if args.fast:
            fast_name = "fast_"
        if args.tiny:
            tiny_name = "tiny_"
            
    file_path = osp.join(args.dataset_root, "cache", f"{args.dataset}_{tiny_name}{fast_name}{name}_obj36.pkl")
    # pdb.set_trace()
    if osp.exists(file_path):
        with open(file_path,'rb') as fp:
            print(f"loading {name} obj_36 cache file...")
            data = pickle.load(fp)
            return data

    data = []
    start_time = time.time()
    print("Start to load Faster-RCNN detected objects from %s" % fname)
    with open(fname) as f:
        reader = csv.DictReader(f, FIELDNAMES, delimiter="\t")
        for i, item in enumerate(tqdm(reader, desc=f'Load object features from {name} tsv file')):

            # 处理一下 "img_id" : COCO_val2014_000000338207 -> 338207
            item["img_id"] = int(item["img_id"].split('_')[-1])
            if img_list is not None and item["img_id"] not in img_list:
                continue

            for key in ['img_h', 'img_w', 'num_boxes']:
                item[key] = int(item[key])
            
            boxes = item['num_boxes']
            decode_config = [
                ('objects_id', (boxes, ), np.int64),
                ('objects_conf', (boxes, ), np.float32),
                ('attrs_id', (boxes, ), np.int64),
                ('attrs_conf', (boxes, ), np.float32),
                ('boxes', (boxes, 4), np.float32),
                ('features', (boxes, -1), np.float32),
            ]
            for key, shape, dtype in decode_config:
                item[key] = np.frombuffer(base64.b64decode(item[key]), dtype=dtype)
                item[key] = item[key].reshape(shape)
                item[key].setflags(write=False)
            
            # TODO：定义一个append的依据：img_id是否被该数据集所需要（e.g. okvqa...）

            

            data.append(item)
            if topk is not None and len(data) == topk:
                break
    elapsed_time = time.time() - start_time
    print("Loaded %d images in file %s in %d seconds." % (len(data), fname, elapsed_time))

    with open(file_path,'wb') as fp:
        print(f"generate {name} obj_36 cache file...")
        pickle.dump(data, fp)

    return data

