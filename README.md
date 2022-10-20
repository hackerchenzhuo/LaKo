# LaKo

![](https://img.shields.io/badge/version-1.0.1-blue)
[![license](https://img.shields.io/github/license/mashape/apistatus.svg?maxAge=2592000)](https://github.com/hackerchenzhuo/LaKo/blob/main/LICENSE)
[![arxiv badge](https://img.shields.io/badge/arXiv-2107.05348-red)](https://arxiv.org/abs/2207.12888)
 - [*LaKo: Knowledge-driven Visual Question Answering via Late Knowledge-to-Text Injection*](https://arxiv.org/abs/2207.12888) 
 

>In this paper, we propose LaKo, a knowledge-driven VQA method via Late Knowledge-to-text Injection. To effectively incorporate an external KG, we transfer triples into text and propose a late injection mechanism. Finally we address VQA as a text generation task with an effective encoder-decoder paradigm. 



## Model Architecture
![Model_architecture](https://github.com/hackerchenzhuo/LaKo/blob/main/figure/github.png)

## Dependencies

- Python 3
- [PyTorch](http://pytorch.org/) (>= 1.6.0)
- [Transformers](http://huggingface.co/transformers/) (**version 3.0.2**)
- [NumPy](http://www.numpy.org/)



### Train


```shell
bash run_okvqa_train.sh
```
or try full training process to get the Attention signal for iterative training

```shell
bash run_okvqa_full.sh
```


### Test

```shell
bash run_okvqa_test.sh
```

**Note**: 
- you can open the `.sh` file for <a href="#Parameter">parameter</a> modification.

### Our code is based on FiD:
- Distilling Knowledge from Reader to Retriever:https://arxiv.org/abs/2012.04584. 
- [Github link to FiD](https://github.com/facebookresearch/FiD)

## Cite:
Please condiser citing this paper if you use the code
```bigquery
@article{DBLP:journals/corr/abs-2207-12888,
  author    = {Zhuo Chen and
               Yufeng Huang and
               Jiaoyan Chen and
               Yuxia Geng and
               Yin Fang and
               Jeff Z. Pan and
               Ningyu Zhang and
               Wen Zhang},
  title     = {LaKo: Knowledge-driven Visual Question Answering via Late Knowledge-to-Text
               Injection},
  journal   = {CoRR},
  volume    = {abs/2207.12888},
  year      = {2022}
}
```
 
