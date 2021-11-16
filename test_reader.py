# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import transformers
import numpy as np
from pathlib import Path
import torch.distributed as dist
from torch.utils.data import DataLoader, SequentialSampler
import os
from tqdm import tqdm
import json
import os.path as osp


import src.slurm
import src.util
from src.options import Options
import src.data
import src.evaluation
import src.model
import nltk.tokenize as tk
import nltk.stem.porter as pt
import time
import pdb


def evaluate(model, dataset, dataloader, tokenizer, opt, dir_path):
    loss, curr_loss = 0.0, 0.0
    model.eval()
    if hasattr(model, "module"):
        model = model.module
    if opt.write_crossattention_scores:
        model.overwrite_forward_crossattention()
        model.reset_score_storage()
    total = 0
    exactmatch = []
    stem_exactmatch = []
    include_exactmatch = []

    if opt.write_results:
        now_time = time.strftime("%m-%d-%H", time.localtime(time.time()))
        if opt.use_fact == "yes":
            fact_para = f"_stream_{opt.stream}_content_{opt.n_context}_"
        else:
            fact_para = ""

        result_name = f"{opt.dataset}_{opt.model_size}_batch_{opt.per_gpu_batch_size}_maxLen_{opt.text_maxlength}{fact_para}{now_time}.json"
        write_path = osp.join(dir_path, 'test_results', result_name)
        result_json = []

    tk_tokenizer = tk.WordPunctTokenizer()
    pt_stemmer = pt.PorterStemmer()
    nl = "\n"
    tl = "\t"
    with torch.no_grad():
        for i, batch in tqdm(enumerate(dataloader), total=len(dataloader), desc=f'Eval'):
            (idx, _, _, context_ids, context_mask) = batch
            if opt.write_crossattention_scores:
                model.reset_score_storage()

            outputs = model.generate(
                input_ids=context_ids.cuda(),
                attention_mask=context_mask.cuda(),
                max_length=50,
            )

            if opt.write_crossattention_scores:
                crossattention_scores = model.get_crossattention_scores(opt, context_ids, tokenizer, context_mask.cuda())
                if opt.ans_attention == "yes":
                    crossattention_scores = crossattention_scores.cpu()
                else:
                    crossattention_scores = torch.softmax(crossattention_scores, dim=-1)

            generated_sents = tokenizer.batch_decode(outputs, skip_special_tokens=True)

            for k, ans in enumerate(generated_sents):
                # ans = tokenizer.decode(o, skip_special_tokens=True)
                example = dataset.data[idx[k]]
                gold = example['answer']
                score = src.evaluation.ems(ans, gold)
                include_score = src.evaluation.includ_ems(ans, gold)
                stem_score = src.evaluation.stem_ems(ans, gold, tk_tokenizer, pt_stemmer, dele_sw=True)

                exactmatch.append(score)
                stem_exactmatch.append(stem_score)
                include_exactmatch.append(include_score)

                if opt.write_results:
                    tmp = {}
                    tmp['question'] = example['question']

                    tmp['img_id'] = example["img_id"]
                    tmp['answer'] = ans
                    tmp['target'] = example['target']
                    tmp['real answers'] = gold
                    tmp['fact'] = example['fact'][:50]
                    tmp['include_score'] = include_score
                    tmp['score'] = score
                    tmp['include_score'] = include_score
                    tmp['stem_score'] = stem_score
                    result_json.append(tmp)

                if opt.write_crossattention_scores:
                    range_num = min(opt.n_context, len(example['fact']))
                    # new attention
                    if opt.ans_attention == "yes":
                        fact_ans_attention = []

                        for fact in example['fact'][:range_num]:

                            fact_sentence = fact["sentence"]
                            fact_ans_attention.append(max(src.evaluation.includ_ems(fact_sentence, gold), src.evaluation.stem_ems(fact_sentence, gold, tk_tokenizer, pt_stemmer, dele_sw=True)))

                        crossattention_scores[k, :range_num] += torch.Tensor(fact_ans_attention)
                        crossattention_scores[k, :range_num] = torch.softmax(crossattention_scores[k, :range_num], dim=-1)
                    # pdb.set_trace()
                    for j in range(range_num):
                        example['fact'][j]['score'] = crossattention_scores[k, j].item()
                total += 1

        if opt.write_results:
            with open(write_path, "w") as f:
                json.dump(result_json, f)

    exactmatch, total = src.util.weighted_average(np.mean(exactmatch), total, opt)
    stem_exactmatch, total = src.util.weighted_average(np.mean(stem_exactmatch), total, opt)
    include_exactmatch, total = src.util.weighted_average(np.mean(include_exactmatch), total, opt)
    return exactmatch, stem_exactmatch, include_exactmatch, total


if __name__ == "__main__":
    options = Options()
    options.add_reader_options()
    options.add_eval_options()
    options.add_optim_options()
    opt = options.parse()

    torch.cuda.set_device(opt.gpu)
    src.slurm.init_distributed_mode(opt)
    src.slurm.init_signal_handler()
    opt.train_batch_size = opt.per_gpu_batch_size * max(1, opt.world_size)

    this_dir = osp.dirname(__file__)
    data_path = osp.abspath(osp.join(this_dir, '..', '..', 'data', 'LaKo'))
    cache_dir = osp.abspath(osp.join(this_dir, '..', '..', 'data', '.cache', 'transformers'))

    opt.device = opt.gpu

    dir_path = osp.join(data_path, opt.checkpoint_dir)

    if opt.write_results:
        result_path = osp.join(dir_path, 'test_results')
        os.makedirs(result_path, exist_ok=True)

    log_dir = osp.join(dir_path, 'run.log')

    logger = src.util.init_logger(opt.is_main, opt.is_distributed, log_dir)

    model_name = 't5-' + opt.model_size

    tokenizer = transformers.T5Tokenizer.from_pretrained(model_name, cache_dir=cache_dir, return_dict=False)

    collator_function = src.data.Collator(opt.text_maxlength, tokenizer, answer_maxlength=opt.answer_maxlength, stream=opt.stream)

    logger.info("data loading ...")

    logger.info("model loading ...")
    model_class = src.model.FiDT5
    opt.model_path = osp.join(data_path, opt.model_path)
    epoch_path = os.path.realpath(opt.model_path)
    model = model_class.from_pretrained(epoch_path)
    # model = model.to(opt.device)
    model = model.cuda()

    eval_data_path = osp.join(data_path, opt.dataset, opt.eval_data)
    with open(eval_data_path, 'r') as fin:
        eval_examples = json.load(fin)
    eval_dataset = src.data.Dataset(eval_examples, opt)

    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(
        eval_dataset,
        sampler=eval_sampler,
        batch_size=opt.per_gpu_batch_size,
        num_workers=10,
        collate_fn=collator_function
    )

    logger.info("Start eval")
    dev_em, dev_stem_em, dev_includ_em, total = evaluate(model, eval_dataset, eval_dataloader, tokenizer, opt, dir_path)
    log = f"evaluation: {100*dev_em:.2f}EM | include: {100*dev_includ_em:.2f}EM| stem: {100*dev_stem_em:.2f}EM , Total number of example {total}|"
    logger.info(log)

    if opt.write_crossattention_scores:
        logger.info("attention save...")

        prefix = opt.eval_data.replace(".json", "")
        write_path = osp.join(dir_path, 'tmp_dir')
        os.makedirs(write_path, exist_ok=True)
        atten = opt.attention_score_style
        if opt.use_last_half_layer_attention == "yes":
            atten = f"last_half_layer_attention_of_{opt.model_size}_with_{opt.attention_score_style}"
        else:
            atten = f"full_attention_of_{opt.model_size}_with_{opt.attention_score_style}"

        save_name = osp.join(write_path, f"{prefix}_{atten}_{opt.version}.json")
        with open(save_name, 'w') as fw:
            json.dump(eval_dataset.data, fw)
        logger.info(f"finish. save to: {save_name}")
