from __future__ import print_function
import os
import os.path as osp
import json
import pickle
import numpy as np
import sys
import re
import pdb
from tqdm import tqdm 

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import utils
import h5py
import torch
from torch.utils.data import Dataset

class Dictionary(object):
    def __init__(self, word2idx=None, idx2word=None):
        if word2idx is None:
            word2idx = {}
        if idx2word is None:
            idx2word = []
        self.word2idx = word2idx
        self.idx2word = idx2word

    @property
    def ntoken(self):
        return len(self.word2idx)

    @property
    def padding_idx(self):
        return len(self.word2idx)

    def tokenize(self, sentence, add_word):
        sentence = sentence.lower()
        sentence = sentence.replace(',', '').replace('?', '').replace('\'s', ' \'s')
        words = sentence.split()
        tokens = []
        if add_word:
            for w in words:
                tokens.append(self.add_word(w))
        else:
            for w in words:
                tokens.append(self.word2idx[w])
        return tokens

    def dump_to_file(self, path):
        pickle.dump([self.word2idx, self.idx2word], open(path, 'wb'))
        print('dictionary dumped to %s' % path)

    @classmethod
    def load_from_file(cls, path):
        print('loading dictionary from %s' % path)
        word2idx, idx2word = pickle.load(open(path, 'rb'))
        d = cls(word2idx, idx2word)
        return d

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)

contractions = {
    "aint": "ain't", "arent": "aren't", "cant": "can't", "couldve":
    "could've", "couldnt": "couldn't", "couldn'tve": "couldn't've",
    "couldnt've": "couldn't've", "didnt": "didn't", "doesnt":
    "doesn't", "dont": "don't", "hadnt": "hadn't", "hadnt've":
    "hadn't've", "hadn'tve": "hadn't've", "hasnt": "hasn't", "havent":
    "haven't", "hed": "he'd", "hed've": "he'd've", "he'dve":
    "he'd've", "hes": "he's", "howd": "how'd", "howll": "how'll",
    "hows": "how's", "Id've": "I'd've", "I'dve": "I'd've", "Im":
    "I'm", "Ive": "I've", "isnt": "isn't", "itd": "it'd", "itd've":
    "it'd've", "it'dve": "it'd've", "itll": "it'll", "let's": "let's",
    "maam": "ma'am", "mightnt": "mightn't", "mightnt've":
    "mightn't've", "mightn'tve": "mightn't've", "mightve": "might've",
    "mustnt": "mustn't", "mustve": "must've", "neednt": "needn't",
    "notve": "not've", "oclock": "o'clock", "oughtnt": "oughtn't",
    "ow's'at": "'ow's'at", "'ows'at": "'ow's'at", "'ow'sat":
    "'ow's'at", "shant": "shan't", "shed've": "she'd've", "she'dve":
    "she'd've", "she's": "she's", "shouldve": "should've", "shouldnt":
    "shouldn't", "shouldnt've": "shouldn't've", "shouldn'tve":
    "shouldn't've", "somebody'd": "somebodyd", "somebodyd've":
    "somebody'd've", "somebody'dve": "somebody'd've", "somebodyll":
    "somebody'll", "somebodys": "somebody's", "someoned": "someone'd",
    "someoned've": "someone'd've", "someone'dve": "someone'd've",
    "someonell": "someone'll", "someones": "someone's", "somethingd":
    "something'd", "somethingd've": "something'd've", "something'dve":
    "something'd've", "somethingll": "something'll", "thats":
    "that's", "thered": "there'd", "thered've": "there'd've",
    "there'dve": "there'd've", "therere": "there're", "theres":
    "there's", "theyd": "they'd", "theyd've": "they'd've", "they'dve":
    "they'd've", "theyll": "they'll", "theyre": "they're", "theyve":
    "they've", "twas": "'twas", "wasnt": "wasn't", "wed've":
    "we'd've", "we'dve": "we'd've", "weve": "we've", "werent":
    "weren't", "whatll": "what'll", "whatre": "what're", "whats":
    "what's", "whatve": "what've", "whens": "when's", "whered":
    "where'd", "wheres": "where's", "whereve": "where've", "whod":
    "who'd", "whod've": "who'd've", "who'dve": "who'd've", "wholl":
    "who'll", "whos": "who's", "whove": "who've", "whyll": "why'll",
    "whyre": "why're", "whys": "why's", "wont": "won't", "wouldve":
    "would've", "wouldnt": "wouldn't", "wouldnt've": "wouldn't've",
    "wouldn'tve": "wouldn't've", "yall": "y'all", "yall'll":
    "y'all'll", "y'allll": "y'all'll", "yall'd've": "y'all'd've",
    "y'alld've": "y'all'd've", "y'all'dve": "y'all'd've", "youd":
    "you'd", "youd've": "you'd've", "you'dve": "you'd've", "youll":
    "you'll", "youre": "you're", "youve": "you've"
}

manual_map = { 'none': '0',
              'zero': '0',
              'one': '1',
              'two': '2',
              'three': '3',
              'four': '4',
              'five': '5',
              'six': '6',
              'seven': '7',
              'eight': '8',
               'nine': '9',
              'ten': '10'}

articles = ['a', 'an', 'the']
period_strip = re.compile("(?!<=\d)(\.)(?!\d)")
comma_strip = re.compile("(\d)(\,)(\d)")
punct = [';', r"/", '[', ']', '"', '{', '}',
                '(', ')', '=', '+', '\\', '_', '-',
                '>', '<', '@', '`', ',', '?', '!']


def get_score(occurences):
    if occurences == 0:
        return 0
    elif occurences == 1:
        return 0.3
    elif occurences == 2:
        return 0.6
    elif occurences == 3:
        return 0.9
    else:
        return 1


def process_punctuation(inText):
    outText = inText
    for p in punct:
        if (p + ' ' in inText or ' ' + p in inText) \
           or (re.search(comma_strip, inText) != None):
            outText = outText.replace(p, '')
        else:
            outText = outText.replace(p, ' ')
    outText = period_strip.sub("", outText, re.UNICODE)
    return outText


def process_digit_article(inText):
    outText = []
    tempText = inText.lower().split()
    for word in tempText:
        word = manual_map.setdefault(word, word)
        if word not in articles:
            outText.append(word)
        else:
            pass
    for wordId, word in enumerate(outText):
        if word in contractions:
            outText[wordId] = contractions[word]
    outText = ' '.join(outText)
    return outText


def multiple_replace(text, wordDict):
    for key in wordDict:
        text = text.replace(key, wordDict[key])
    return text


def preprocess_answer(answer):
    answer = process_digit_article(process_punctuation(answer))
    answer = answer.replace(',', '')
    return answer


def filter_answers(answers_dset, dataset, min_occurence):
    """This will change the answer to preprocessed version
    """
    
    
    occurence = {}
    print("filter answers:")

    for ans_entry in tqdm(answers_dset, desc=f'deal {dataset} answer set'):
        answers = ans_entry['answers']
        if dataset == "vqa2.0":
            gtruths = [ans_entry['multiple_choice_answer']]
        elif dataset == "okvqa":
            gtruths = list(set([ans["answer"] for ans in answers]))
        for gtruth in gtruths:
            gtruth = preprocess_answer(gtruth)
            if gtruth not in occurence:
                occurence[gtruth] = set()
            occurence[gtruth].add(ans_entry['question_id'])

    for answer in list(occurence.keys()):
        if len(occurence[answer]) < min_occurence:
            occurence.pop(answer)

    print('Num of answers that appear >= %d times: %d' % (
        min_occurence, len(occurence)))
    return occurence


def create_ans2label(name, args, answers, min_occurence):
    """Note that this will also create label2ans.pkl at the same time
    occurence: dict {answer -> whatever}
    name: prefix of the output file
    cache_root: str
    """
    cache_root=osp.join(args.data_root, args.dataset, "cache", str(int(args.min_occurence)))
    cache_file_ans2label = os.path.join(cache_root, name + '_ans2label.json')
    cache_file_label2ans = os.path.join(cache_root, name + '_label2ans.json')

    if osp.exists(cache_file_ans2label) and osp.exists(cache_file_label2ans):
        print(f"exist: {cache_file_ans2label} and {cache_file_label2ans} ")
        with open(cache_file_ans2label, 'r') as fp:
            ans2label = json.load(fp)
    else:
        occurence = filter_answers(answers, args.dataset, min_occurence)
        ans2label = {}
        label2ans = []
        label = 0
        for answer in occurence.keys():
            label2ans.append(answer)
            ans2label[answer] = label
            label += 1

        utils.create_dir(cache_root)

        with open(cache_file_ans2label, 'w') as fp:
            json.dump(ans2label, fp)
        with open(cache_file_label2ans, 'w') as fp:
            json.dump(label2ans, fp)
        
        # pickle.dump(ans2label, open(cache_file_ans2label, 'wb'))
        # pickle.dump(label2ans, open(cache_file_label2ans, 'wb'))

    return ans2label


def compute_target(answers_dset, ans2label, id2question, name, args):
    """Augment answers_dset with soft score as label
    ***answers_dset should be preprocessed***
    Write result into a cache file
    """
    """
    A VQA data example in json file:
        {
            "answer_type": "other",
            "img_id": "COCO_train2014_000000458752",
            "label": {
                "net": 1
            },
            "question_id": 458752000,
            "question_type": "what is this",
            "sent": "What is this photo taken looking through?"
        }
    """
    # ref: https://github.com/airsplay/lxmert/blob/0db1182b9030da3ce41f17717cc628e1cd0a95d5/src/tasks/vqa_data.py#L33 读取数据
    # ref: https://github.com/airsplay/lxmert/blob/0db1182b9030da3ce41f17717cc628e1cd0a95d5/src/tasks/vqa_data.py#L99 top几数据
    cache_root=osp.join(args.data_root, args.dataset, "cache", str(int(args.min_occurence)))
    cache_file = os.path.join(cache_root, name + '.json')

    all_ans = set()

    # if osp.exists(cache_file):
    #     print(f"exist: {cache_file}")
    #     return 

    target = []
    for ans_entry in tqdm(answers_dset, desc=f'deal {name} json file'):
        # pdb.set_trace()
        answers = ans_entry['answers']
        answer_count = {}
        for answer in answers:
            answer_ = preprocess_answer(answer['answer'])
            answer_count[answer_] = answer_count.get(answer_, 0) + 1

        labels = {}
        scores = []
        for answer in answer_count:
            if answer not in ans2label.keys():
                continue
            all_ans.add(answer)
            labels[answer] = get_score(answer_count[answer])
        # if len(labels) > 0 and max(labels.values()) == 1: # 

        # VQA2.0:
        # 标准答案必须出现过n次以上才统计： 371950
        # 全部统计： 443757
        # pdb.set_trace()
        target.append({
            'answer_type': ans_entry['answer_type'],
            'img_id': ans_entry['image_id'],
            'label': labels,
            'question_id': ans_entry['question_id'],
            "question_type": ans_entry['question_type'],
            "sent": id2question[str(ans_entry['question_id'])]
        })
            
    # pdb.set_trace()

    utils.create_dir(cache_root)
    

    # pickle.dump(target, open(cache_file, 'wb'))
    
    with open(cache_file, 'w') as fp:
        json.dump(target, fp)


    return target, all_ans


def get_answer(qid, answers):
    for ans in answers:
        if ans['question_id'] == qid:
            return ans


def get_question(qid, questions):
    for question in questions:
        if question['question_id'] == qid:
            return question

