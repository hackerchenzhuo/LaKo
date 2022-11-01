import os.path as osp
import numpy as np
import random
import torch
from easydict import EasyDict as edict
import argparse
import pdb


class cfg():
    def __init__(self):
        self.this_dir = osp.dirname(__file__)
        # change
        self.data_root = osp.abspath(osp.join(self.this_dir, '..', '..', 'data', 'KPVQA'))

        # TODO: add some static variable  (The frequency of change is low)
        self.common_cache_path = osp.join(self.data_root, 'common_data')
        self.w2v_dim = 300
        self.qs_cache_w2v = f'glove.{self.w2v_dim}d.txt.npy'

        self.TRAIN = edict()
        self.TRAIN.lr = 2e-5  # default Adam lr 1e-3
        self.TRAIN.lr_decay_step = 2
        self.TRAIN.lr_decay_rate = .25

    def get_args(self):
        parser = argparse.ArgumentParser()
        # base
        parser.add_argument('--gpu', default=0, type=int)
        parser.add_argument('--num_workers', default=4, type=int)
        parser.add_argument('--batch_size', default=128, type=int)
        parser.add_argument('--lr', default=2e-5, type=float)
        parser.add_argument('--epoch', default=100, type=int)
        parser.add_argument("--save_model", action='store_const', default=False, const=True)

        # Model Loading
        # 优先级： load >
        parser.add_argument('--load', type=str, default=None,
                            help='Load the model (usually the fine-tuned model).')
        # parser.add_argument('--loadLXMERT', dest='load_lxmert', type=str, default=None,
        #                     help='Load the pre-trained LXMERT model.')
        # parser.add_argument('--loadLXMERTQA', dest='load_lxmert_qa', type=str, default=None,
        #                     help='Load the pre-trained LXMERT model with QA answer head.')

        parser.add_argument("--fromScratch", dest='from_scratch', action='store_const', default=False, const=True,
                            help='If none of the --load, --loadLXMERT, --loadLXMERTQA is set, '
                            'the model would be trained from scratch. If --fromScratch is'
                            ' not specified, the model would load BERT-pre-trained weights by'
                            ' default. ')
        parser.add_argument("--finetune", action='store_const', default=False, const=True)
        # parser.add_argument("--finetune_part", action='store_const', default=False, const=True)

        # torthlight
        parser.add_argument("--no_tensorboard", default=False, action="store_true")
        parser.add_argument("--exp_name", default="test", type=str, help="Experiment name")
        parser.add_argument("--dump_path", default="dump/", type=str, help="Experiment dump path")
        parser.add_argument("--exp_id", default="001", type=str, help="Experiment ID")
        parser.add_argument("--random_seed", default=1104, type=int)

        # add some dynamic variable
        parser.add_argument("--train", default="train", type=str)
        parser.add_argument("--valid", default="valid", type=str)
        parser.add_argument("--test", default="valid", type=str)
        parser.add_argument("--vqa_test", action='store_const', default=False, const=True)
        parser.add_argument("--dataset", default="okvqa", type=str, choices=["okvqa", "vqa2.0", "fvqa"])
        parser.add_argument("--dictionary", default="qs_dictionary.pkl", type=str)
        parser.add_argument("--patience", default=10, type=int)
        parser.add_argument("--warmup_ratio", default=0, type=float)
        parser.add_argument("--min_occurence", default=10, type=float)

        parser.add_argument("--fast", action='store_const', default=False, const=True)
        parser.add_argument("--tiny", action='store_const', default=False, const=True)
        parser.add_argument("--output_attention", action='store_const', default=False, const=True)

        # add some prompt variable
        # 0: no prompt
        # 1: <question + answer> image
        #
        parser.add_argument("--prompt", default=0, type=int)
        parser.add_argument("--detail_dump_result", action='store_const', default=False, const=True)

        # knowledge and question stream:
        parser.add_argument("--two_stream", action='store_const', default=False, const=True)
        # parser.add_argument("--one_stream", action='store_const', default=False, const=True)
        parser.add_argument("--split_segment", action='store_const', default=False, const=True)

        args = parser.parse_args()
        return args

    def update_train_configs(self, args):
        self.gpu = args.gpu
        self.batch_size = args.batch_size
        self.epoch = args.epoch
        self.save_model = args.save_model
        self.vqa_test = args.vqa_test
        self.no_tensorboard = args.no_tensorboard
        self.exp_name = args.exp_name
        self.dump_path = args.dump_path
        # # change dump path
        # self.dump_path = osp.join(self.data_root, "dump")

        self.exp_id = args.exp_id
        self.random_seed = args.random_seed
        self.num_workers = args.num_workers
        self.TRAIN.lr = float(args.lr)

        # update some dynamic variable
        self.dataset = args.dataset
        self.dictionary = args.dictionary
        self.valid = args.valid
        self.train = args.train
        self.test = args.test
        self.patience = args.patience
        self.fast = args.fast
        self.tiny = args.tiny
        self.warmup_ratio = args.warmup_ratio
        self.prompt = args.prompt
        self.detail_dump_result = args.detail_dump_result
        self.min_occurence = args.min_occurence

        self.two_stream = args.two_stream
        # self.one_stream = args.one_stream
        self.split_segment = args.split_segment
        self.output_attention = args.output_attention
        # model load:
        self.load = args.load
        # self.loadLXMERT = args.loadLXMERT
        # self.loadLXMERTQA = args.loadLXMERTQA
        self.from_scratch = args.from_scratch
        self.finetune = args.finetune
        # self.finetune_part = args.finetune_part

        # add some constraint for parameters
        # e.g. cannot save and test at the same time
        # if self.load != "":
        #     self.save_model = False

        assert not (self.save_model and self.vqa_test)
        assert not (self.finetune and self.from_scratch)
        assert self.min_occurence > 0
        # if self.dataset == "vqa2.0":
        #     self.min_occurence = 9
        # if self.dataset == "okvqa":
        #     assert self.min_occurence in [1, 3, 5, 10]

        # if self.prompt != 0:
        #     assert self.two_stream or self.one_stream
        #     # cannot be the same
        #     assert not (self.two_stream and self.one_stream)
