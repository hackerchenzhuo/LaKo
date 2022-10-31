from __future__ import print_function
from config import cfg
import os.path as osp
import pdb
import tqdm
import os
import sys
import json
import numpy as np
from model import Vector

from data.data_init import Dictionary, filter_answers, create_ans2label, compute_target, compute_target


class Runner:
    def __init__(self, args):
        self.args = args
        self.project_root = args.data_root
        self.dataroot = osp.join(args.data_root, args.dataset)

        self.dictionary = args.dictionary
        assert args.dataset in ["vqa2.0", "okvqa"], "args.dataset have to be on [vqa2.0,okvqa]"

        # vqa
        if args.dataset == 'vqa2.0':
            self.train_answer_file = 'v2_mscoco_train2014_annotations.json'
            self.val_answer_file = 'v2_mscoco_val2014_annotations.json'

            self.train_question_file = 'v2_OpenEnded_mscoco_train2014_questions.json'
            self.val_question_file = 'v2_OpenEnded_mscoco_val2014_questions.json'
            self.test_question_file = 'v2_OpenEnded_mscoco_test2015_questions.json'
            self.test_dev_question_file = 'v2_OpenEnded_mscoco_test-dev2015_questions.json'
            self.files = [self.train_question_file, self.val_question_file, self.test_question_file, self.test_dev_question_file]

        # okvqa
        elif args.dataset == 'okvqa':
            self.train_answer_file = 'mscoco_train2014_annotations.json'
            self.val_answer_file = 'mscoco_val2014_annotations.json'

            self.train_question_file = 'OpenEnded_mscoco_train2014_questions.json'
            self.val_question_file = 'OpenEnded_mscoco_val2014_questions.json'
            self.files = [self.train_question_file, self.val_question_file]

        elif args.dataset == "fvqa":
            pass

        self.train_answers = json.load(open(osp.join(self.dataroot, "common", self.train_answer_file)))['annotations']
        self.val_answers = json.load(open(osp.join(self.dataroot, "common", self.val_answer_file)))['annotations']
        # self.train_questions = json.load(open(osp.join(self.dataroot, "common", self.train_question_file)))['questions']
        # self.val_questions = json.load(open(sosp.join(self.dataroot, "common", self.val_question_file)))['questions']

        self.answers = self.train_answers + self.val_answers

    def id_to_question(self):
        cache_root = osp.join(self.args.data_root, self.args.dataset, "cache")
        cache_file = os.path.join(cache_root, 'id2question.json')

        if osp.exists(cache_file):
            print(f"exist: {cache_file}")
            with open(cache_file, 'r') as fp:
                self.id2question = json.load(fp)

            return

        self.id2question = {}
        for path in self.files:
            question_path = os.path.join(self.dataroot, "common", path)
            qs = json.load(open(question_path))['questions']
            for item in qs:
                self.id2question[item['question_id']] = item['question']

        with open(cache_file, 'w') as fp:
            json.dump(self.id2question, fp)

    def create_dictionary(self):
        self.word2vec = Vector(self.args.common_cache_path)
        dictionary = Dictionary()
        questions = []
        for ques in self.id2question.values():
            dictionary.tokenize(ques, True)
        return dictionary

    def get_dictionary(self):
        dictionary_path = os.path.join(self.dataroot, "cache", self.dictionary)

        if osp.exists(dictionary_path):
            d = Dictionary.load_from_file(dictionary_path)
        else:
            d = self.create_dictionary()
            d.dump_to_file(dictionary_path)

        print("qs dictionary done!")
        qs_cache_w2v = os.path.join(self.dataroot, "cache", self.args.qs_cache_w2v)

        if not osp.exists(qs_cache_w2v):
            weights, word2emb = self.create_glove_embedding_init(d.idx2word)
            np.save(qs_cache_w2v, weights)
        print("w2v done!")

    def create_glove_embedding_init(self, idx2word):
        word2emb = {}
        weights = np.zeros((len(idx2word), self.args.w2v_dim), dtype=np.float32)
        for idx, word in enumerate(idx2word):
            if self.word2vec.check(word):
                weights[idx] = self.word2vec[word]
            else:
                continue

        return weights, word2emb

    def compute_softscore(self, min_occurence):
        ans2label = create_ans2label(f'trainval', self.args, self.answers, min_occurence)

        _, train_all_ans = compute_target(self.train_answers, ans2label, self.id2question, f'train', self.args)
        _, val_all_ans = compute_target(self.val_answers, ans2label, self.id2question, f'valid', self.args)
        print(f"train answer set len:{len(train_all_ans)}")
        print(f"val answer set len:{len(val_all_ans)}")

        print(f"answer set & len:{len(train_all_ans & val_all_ans)}")
        print(f"answer set | len:{len(train_all_ans | val_all_ans)}")
        print(f"ans2label len:{len(ans2label)}")


if __name__ == '__main__':
    cfg = cfg()
    args = cfg.get_args()
    ############## okvqa ##############
    # min occ:10 -> 896 [train: 896 test: 892]
    # val answer set len:892
    # answer set & len:892
    # answer set | len:896
    # ans2label len:896

    # min occ:5 -> 1858 [train: 1854 test: 1785]
    # min occ:3 -> 3065 [train: 3016 test: 2678]
    # min occ:1 -> 15038 [train: 11507 test: 6914]

    # args.dataset = "okvqa"
    # cfg.update_train_configs(args)
    # runner = Runner(cfg)
    # # create_dictionary
    # runner.id_to_question()
    # runner.get_dictionary()
    # # compute_softscore
    # min_occurence = cfg.min_occurence
    # runner.compute_softscore(min_occurence)

    ############## vqa2.0 ##############
    # train answer set len:3126
    # val answer set len:3122
    # answer set & len:3119
    # answer set | len:3129
    # ans2label len:3129

    args.dataset = "vqa2.0"
    cfg.update_train_configs(args)
    runner = Runner(cfg)
    runner.id_to_question()
    # create_dictionary
    runner.get_dictionary()
    # compute_softscore
    min_occurence = cfg.min_occurence  # 9
    # min_occurence = 1
    print("min_occurence:", min_occurence)
    runner.compute_softscore(min_occurence)

    ############## fvqa ##############
    # min_occurence = 3 # TOP 285 answer
