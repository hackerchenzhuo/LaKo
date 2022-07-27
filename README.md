### This repository contains code for:
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

