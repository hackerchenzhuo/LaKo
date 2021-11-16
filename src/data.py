# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import random
import json
import numpy as np
import pdb


class Dataset(torch.utils.data.Dataset):
    def __init__(self,
                 data,
                 opt,
                 fact_use_way="concate",
                 question_prefix='question:',  # question
                 caption_prefix='context:',  # img caption
                 fact_prefix='fact:'):  # knowledge
        self.data = data
        self.n_context = opt.n_context
        self.question_prefix = question_prefix
        self.caption_prefix = caption_prefix
        self.fact_prefix = fact_prefix
        #
        self.fact_use_way = opt.fact_use_way
        self.use_fact = opt.use_fact

    def __len__(self):
        return len(self.data)

    def get_target(self, example):
        if 'target' in example:
            target = example['target']
            return target + ' </s>'
        elif 'answers' in example:
            return random.choice(example['answers']) + ' </s>'
        else:
            return None

    def __getitem__(self, index):
        example = self.data[index]
        question = self.question_prefix + " " + example['question']
        caption = self.caption_prefix + " " + example["caption"]
        target = self.get_target(example)

        answer = example["answer"]
        scores = None
        fact = None
        #
        if self.use_fact == "yes":
            contexts = example['fact'][:self.n_context]
            fact_list = [c['sentence'] for c in contexts]
            if self.fact_use_way == "concate":
                fact = ' '.join(fact_list)
                fact = self.fact_prefix + " " + fact + " "
            else:
                fact = fact_list

            if 'score' in contexts[0]:
                scores = [float(c['score']) for c in contexts]
                scores = torch.tensor(scores)

            # TODO(egrave): do we want to keep this?
            if len(contexts) == 0:
                contexts = [self.fact_prefix]

        # pdb.set_trace()
        return {
            'index': index,
            'question': question,
            'caption': caption,
            'target': target,
            'answer': answer,
            'fact': fact,
            'score': scores
        }

    def sort_data(self):
        pass

    def get_example(self, index):
        return self.data[index]


def encode_passages(batch_text_passages, tokenizer, max_length):
    passage_ids, passage_masks = [], []
    for k, text_passages in enumerate(batch_text_passages):
        p = tokenizer.batch_encode_plus(
            text_passages,
            max_length=max_length,
            pad_to_max_length=True,
            return_tensors='pt',
            truncation=True
        )

        passage_ids.append(p['input_ids'][None])
        passage_masks.append(p['attention_mask'][None])

    passage_ids = torch.cat(passage_ids, dim=0)
    passage_masks = torch.cat(passage_masks, dim=0)
    return passage_ids, passage_masks.bool()


class Collator(object):
    def __init__(self, text_maxlength, tokenizer, answer_maxlength=20, stream=2):
        self.tokenizer = tokenizer
        self.text_maxlength = text_maxlength
        self.answer_maxlength = answer_maxlength
        self.stream = stream
        # self.psg_len = []

    def __call__(self, batch):
        # assert(batch[0]['target'] != None)
        index = torch.tensor([ex['index'] for ex in batch])
        target = [ex['target'] for ex in batch]
        target = self.tokenizer.batch_encode_plus(
            target,
            max_length=self.answer_maxlength if self.answer_maxlength > 0 else None,
            pad_to_max_length=True,
            return_tensors='pt',
            truncation=True if self.answer_maxlength > 0 else False,
        )
        target_ids = target["input_ids"]
        target_mask = target["attention_mask"].bool()
        target_ids = target_ids.masked_fill(~target_mask, -100)

        def append_question(example):
            if example['fact'] is None:
                # return [example['question']]
                return [example['question'] + " " + example['caption']]
            elif type(example['fact']) is str:
                if self.stream == 1:
                    return [example['question'] + " " + example['caption'] + " " + example['fact']]
                else:
                    return [example['question'] + " " + example['caption'], example['fact']]
             # TODO: fact separate
            else:
                pass

        text_passages = [append_question(example) for example in batch]

        passage_ids, passage_masks = encode_passages(text_passages,
                                                     self.tokenizer,
                                                     self.text_maxlength)
        return (index, target_ids, target_mask, passage_ids, passage_masks)


def load_data(data_path=None, global_rank=-1, world_size=-1):
    assert data_path
    if data_path.endswith('.jsonl'):
        data = open(data_path, 'r')
    elif data_path.endswith('.json'):
        with open(data_path, 'r') as fin:
            data = json.load(fin)
    examples = []
    for k, example in enumerate(data):
        if global_rank > -1 and not k % world_size == global_rank:
            continue
        if data_path is not None and data_path.endswith('.jsonl'):
            example = json.loads(example)
        if not 'id' in example:
            example['id'] = k

        for c in example['fact']:
            if not 'score' in c:
                c['score'] = 1.0 / (k + 1)
        examples.append(example)

    if data_path is not None and data_path.endswith('.jsonl'):
        data.close()

    return examples


class RetrieverCollator(object):
    def __init__(self, tokenizer, passage_maxlength=140, question_maxlength=140):
        self.tokenizer = tokenizer
        self.passage_maxlength = passage_maxlength
        self.question_maxlength = question_maxlength

    def __call__(self, batch):
        index = torch.tensor([ex['index'] for ex in batch])

        question = [ex['question'] + " " + ex['caption'] for ex in batch]
        question = self.tokenizer.batch_encode_plus(
            question,
            pad_to_max_length=True,
            return_tensors="pt",
            max_length=self.question_maxlength,
            truncation=True
        )
        question_ids = question['input_ids']
        question_mask = question['attention_mask'].bool()

        if batch[0]['score'] is None or batch[0]['fact'] is None:
            return index, question_ids, question_mask, None, None, None

        scores = [ex['score'] for ex in batch]
        scores = torch.stack(scores, dim=0)

        passages = [ex['fact'] for ex in batch]
        passage_ids, passage_masks = encode_passages(
            passages,
            self.tokenizer,
            self.passage_maxlength
        )

        return (index, question_ids, question_mask, passage_ids, passage_masks, scores)


class TextDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        example = self.data[index]
        # contexts = example['fact']
        fact = example['sentence']
        ids = int(example['id'])
        return {"fact": fact, "id": ids}


class TextCollator(object):
    def __init__(self, tokenizer, maxlength=100):
        self.tokenizer = tokenizer
        self.maxlength = maxlength

    def __call__(self, batch):

        fact_ids = torch.tensor([ex['id'] for ex in batch])
        fact = [ex['fact'] for ex in batch]

        encoded_batch = self.tokenizer.batch_encode_plus(
            fact,
            pad_to_max_length=True,
            return_tensors="pt",
            max_length=self.maxlength,
            truncation=True
        )
        text_ids = encoded_batch['input_ids']
        text_mask = encoded_batch['attention_mask'].bool()
        # pdb.set_trace()
        return fact_ids, text_ids, text_mask
