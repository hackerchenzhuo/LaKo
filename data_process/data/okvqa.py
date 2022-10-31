## ref: https://github.com/s3vqa/s3vqa.github.io/blob/main/code/preprocessing/data.py

import json
import string
import re
from tqdm import tqdm
from collections import Counter
from nltk import word_tokenize
import nltk
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize 
from collections import OrderedDict
from nltk.stem import PorterStemmer
ps = PorterStemmer()
span = re.compile('\n')
import torch
from transformers import BertTokenizer, BertModel
from tqdm import tqdm
import json, os, sys
from torch.utils.data import Dataset
import pickle
import math
from transformers import AutoTokenizer
 
class OKVQA(Dataset):
 
    def __init__(self, path, file):
        self.max_sequence_length_question = 32
        self.max_sequence_length_hypernym = 8
        self.max_sequence_length_hyponym = 8
        self.max_hyponyms = 32
        self.max_hypernyms = 32
        self.min_isadb_score = -10000000000.0
        self.default_hypo_score = 0.5
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased') #AutoTokenizer.from_pretrained("SpanBERT/spanbert-base-cased")  
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.embedding = self.bert.get_input_embeddings()
 
        if os.path.exists(path + '.pickle'):
            self.data = pickle.load(open(path + '.pickle','rb'))
        else:
            self.data = self.preprocess(file)
            with open(path + '.pickle','wb') as f:
                pickle.dump(self.data,f)
 
    def preprocess(self, file):
        list_question_input_ids = []
        list_hypernym_input_ids = []
        list_hyponym_input_ids = []
        list_detected_hyponym_input_ids = []
        list_detected_hyponym_mask = []
        list_detected_hyponym_embeddings = []
        list_detected_hyponym_scores = []
        list_score_isadb = []
        list_label_i = []
        list_label_j = []   
        list_label_hypo_input_ids = []
        list_label_hypo = []
        
        list_new_gquestion = []
        list_new_gi = []
        list_new_gj = []
        list_new_questions = []
        list_new_is = []
        list_new_js = []
 
        detections_dict = {}
        # index = 0
        list_objects_input_ids = []
        
        list_hypo_scores = []
    
        with open(file, encoding="utf-8") as f:
            data = json.load(f)
            # data = data[0:50]
            for q_id, q in tqdm(data.items()):
                if q["hyponym"] in q["detections"]:
                    question = q["question"]
                    label_hyper = q["hypernym"]
                    label_hypo = q["hyponym"]
                    hypernymy = q["hypernymy_relations"]   ##not needed anymore
                    detected_hyponyms = q["detections"]

                    question_encoded = self.tokenizer.encode(question,max_length=self.max_sequence_length_question,
                                                             pad_to_max_length=True,truncation=False)
                    label_hyper_encoded = self.tokenizer.encode(label_hyper,pad_to_max_length=False,
                                                                add_special_tokens=False)   

                    def find_sub_list(sl,l):
                        sll=len(sl)
                        for ind in (i for i,e in enumerate(l) if e==sl[0]):
                            if l[ind:ind+sll]==sl:
                                return ind,ind+sll-1
                    try:
                        label_i,label_j = find_sub_list(label_hyper_encoded, question_encoded)
                    except:
                        print(label_hyper,question)
                        continue


                    question = q["question"]
                    label_hyper = q["hypernym"]
                    label_hypo = q["hyponym"]
                    detected_hyponyms = q["detections"]
                    new_questions = []
                    new_is, new_js = [], []
                    span = re.compile(label_hyper, re.IGNORECASE)
                    hypo = label_hypo
                    if ' ' == label_hyper[0]:
                        hypo = ' ' + label_hypo
                    if ' ' == label_hyper[-1]:
                        hypo = label_hypo + ' '
                    new_question = span.sub(hypo, question)
                    new_question_encoded = self.tokenizer.encode(new_question,
                                                                 max_length=self.max_sequence_length_question,
                                                                 pad_to_max_length=True,truncation=False)
                    hypo_encoded = self.tokenizer.encode(hypo, pad_to_max_length=False,
                                                                add_special_tokens=False)   
                    def find_sub_list(sl,l):
                        sll=len(sl)
                        for ind in (i for i,e in enumerate(l) if e==sl[0]):
                            if l[ind:ind+sll]==sl:
                                return ind,ind+sll-1
                    try:
                        new_i, new_j = find_sub_list(hypo_encoded, new_question_encoded)
                        list_new_gquestion.append(new_question_encoded)
                        list_new_gi.append(new_i)
                        list_new_gj.append(new_j)
                    except:
                        print(hypo, new_question)
                        continue
                    for detected_hypo in detected_hyponyms:
                        span = re.compile(label_hyper, re.IGNORECASE)
                        hypo = detected_hypo
                        if ' ' == label_hyper[0]:
                            hypo = ' ' + detected_hypo
                        if ' ' == label_hyper[-1]:
                            hypo = detected_hypo + ' '
                        new_question = span.sub(hypo, question)
                        new_question_encoded = self.tokenizer.encode(new_question,
                                                                     max_length=self.max_sequence_length_question,
                                                                     pad_to_max_length=True,truncation=False)
                        hypo_encoded = self.tokenizer.encode(hypo, pad_to_max_length=False,
                                                                    add_special_tokens=False)   
                        def find_sub_list(sl,l):
                            sll=len(sl)
                            for ind in (i for i,e in enumerate(l) if e==sl[0]):
                                if l[ind:ind+sll]==sl:
                                    return ind,ind+sll-1
                        try:
                            new_i, new_j = find_sub_list(hypo_encoded, new_question_encoded)
                        except:
                            print(hypo, new_question)
                            continue
                        new_questions.append(new_question_encoded)
                        new_is.append(new_i)
                        new_js.append(new_j)
                    while(len(new_questions)<self.max_hyponyms):
                        new_is.append(0.0)
                        new_js.append(0.0)
                        new_questions.append(torch.zeros(self.max_sequence_length_question, dtype=torch.int64))
                    list_new_questions.append(new_questions)
                    list_new_is.append(new_is)
                    list_new_js.append(new_js)


                    list_label_i.append(label_i)
                    list_label_j.append(label_j)
                    list_question_input_ids.append(question_encoded)

                    label_hypo_input_ids = self.tokenizer.encode(label_hypo,max_length=self.max_sequence_length_hyponym,
                                                                 add_special_tokens=False,pad_to_max_length=True,
                                                                 truncation=False)
                    list_label_hypo_input_ids.append(label_hypo_input_ids)

                    scores = {}
                    for hypo,hypers in hypernymy.items():
                        score = 0.0
                        hypernyms = set()
                        for hyper in hypers:
                            hn = hyper["hypernym"]
                            if label_hyper.count(hn) > 0 and not hn in hypernyms:
                                score += math.exp(hyper["score"])
                            hypernyms.add(hn)
                        scores[hypo] = score

                    detected_hyponyms = q["detections"]
                    o2hs = q['o2hs']
                    hypo_scores = []
                    for detected_hypo in detected_hyponyms:
                        hypo_scores.append(o2hs[detected_hypo.lower()])
                    while(len(hypo_scores) < self.max_hyponyms):
                        hypo_scores.append(0.0)
                    list_hypo_scores.append(hypo_scores)

                    detected_hyponym_input_ids = []
                    detected_hyponym_mask = []
                    detected_hyponym_embeddings = []
                    detected_hyponym_scores = []
                    hypo_id = -1
                    index = 0
                    for detected_hypo in detected_hyponyms:
                        if detected_hypo == label_hypo:
                            hypo_id = index
                        detected_hypo_encoded = self.tokenizer(detected_hypo,max_length=self.max_sequence_length_hyponym,
                                                               pad_to_max_length=True,truncation=False,
                                                               add_special_tokens=False,return_attention_mask=True)
                        detected_hyponym_input_ids.append(detected_hypo_encoded["input_ids"])
                        detected_hyponym_mask.append(detected_hypo_encoded["attention_mask"])
                        detected_hypo_embedding = torch.mean(self.embedding(torch.tensor
                                                                        (self.tokenizer.encode
                                                                        (detected_hypo,pad_to_max_length=False,
                                                                        add_special_tokens=False),dtype=torch.int64)),dim=0)
                        detected_hyponym_embeddings.append(detected_hypo_embedding)
                        if detected_hypo in scores:
                            detected_hyponym_scores.append(scores[detected_hypo])
                        else:
                            detected_hyponym_scores.append(self.default_hypo_score)
                        index += 1

                    while(len(detected_hyponym_input_ids) < self.max_hyponyms):
                        detected_hyponym_input_ids.append(torch.zeros(self.max_sequence_length_hyponym, dtype=torch.int64))
                        detected_hyponym_mask.append(torch.zeros(self.max_sequence_length_hyponym, dtype=torch.int64))
                        detected_hyponym_embeddings.append(torch.zeros(768, dtype=torch.float64))
                        detected_hyponym_scores.append(0.0)

                    detected_hyponym_embeddings = torch.stack(detected_hyponym_embeddings,dim=0)
                    list_detected_hyponym_input_ids.append(detected_hyponym_input_ids)
                    list_detected_hyponym_mask.append(detected_hyponym_mask)
                    list_detected_hyponym_embeddings.append(detected_hyponym_embeddings)
                    list_label_hypo.append(hypo_id)
                    list_detected_hyponym_scores.append(detected_hyponym_scores)

                    hypernym_input_ids = []
                    hyponym_input_ids = []
                    score_isadb = []

                
        list_question_input_ids = torch.tensor(list_question_input_ids, dtype=torch.int64)
        list_detected_hyponym_input_ids = torch.tensor(list_detected_hyponym_input_ids, dtype=torch.int64)
        list_detected_hyponym_mask = torch.tensor(list_detected_hyponym_mask, dtype=torch.int64)
        list_detected_hyponym_embeddings = torch.stack(list_detected_hyponym_embeddings, dim=0)
        list_detected_hyponym_scores = torch.tensor(list_detected_hyponym_scores, dtype=torch.float64)
        list_label_i = torch.tensor(list_label_i, dtype=torch.int64)
        list_label_j = torch.tensor(list_label_j, dtype=torch.int64)
        list_label_hypo = torch.tensor(list_label_hypo, dtype=torch.int64)
        list_label_hypo_input_ids = torch.tensor(list_label_hypo_input_ids, dtype=torch.int64)
        
        list_new_gquestion = torch.tensor(list_new_gquestion, dtype=torch.int64)
        list_new_gi = torch.tensor(list_new_gi, dtype=torch.int64)
        list_new_gj = torch.tensor(list_new_gj, dtype=torch.int64)
        
        list_new_questions = torch.tensor(list_new_questions, dtype=torch.int64)
        list_new_is = torch.tensor(list_new_is, dtype=torch.int64)
        list_new_js = torch.tensor(list_new_js, dtype=torch.int64)
        
        list_hypo_scores = torch.tensor(list_hypo_scores, dtype=torch.int64)

        dataset = {
            'question_input_ids': list_question_input_ids,
            'detected_hyponym_input_ids' : list_detected_hyponym_input_ids,
            'detected_hyponym_mask' : list_detected_hyponym_mask,
            'detected_hyponym_embeddings' : list_detected_hyponym_embeddings,
            'detected_hyponym_scores' : list_detected_hyponym_scores,
            'label_i': list_label_i,
            'label_j': list_label_j,
            'label_hypo': list_label_hypo,
            'label_hypo_input_ids' : list_label_hypo_input_ids,
            'new_gquestion' : list_new_gquestion,
            'new_gi' : list_new_gi,
            'new_gj' : list_new_gj,
            'new_questions' : list_new_questions,
            'new_is' : list_new_is,
            'new_js' : list_new_js,
            'hypo_scores' : list_hypo_scores,
        }
        print("Total obects = ", index)
        return dataset
 
 
    def __len__(self):
        return self.data['question_input_ids'].shape[0]
 
    def __getitem__(self, id):
        return {
            'question_input_ids': self.data['question_input_ids'][id],
            'detected_hyponym_input_ids': self.data['detected_hyponym_input_ids'][id],
            'detected_hyponym_mask': self.data['detected_hyponym_mask'][id],
            'detected_hyponym_embeddings': self.data['detected_hyponym_embeddings'][id],
            'detected_hyponym_scores' : self.data['detected_hyponym_scores'][id],
            'label_i': self.data['label_i'][id],
            'label_j': self.data['label_j'][id],
            'label_hypo': self.data['label_hypo'][id],
            'label_hypo_input_ids' :self.data['label_hypo_input_ids'][id],
            'new_gquestion' : self.data['new_gquestion'][id],
            'new_gi' : self.data['new_gi'][id],
            'new_gj' : self.data['new_gj'][id],
            'new_questions' : self.data['new_questions'][id],
            'new_is': self.data['new_is'][id],
            'new_js' : self.data['new_js'][id],
            'hypo_scores' : self.data['hypo_scores'][id],
        }