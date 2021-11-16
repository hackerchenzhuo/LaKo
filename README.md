This repository contains code for:
 - Knowledge-driven Visual Question Answering via Late Knowledge-to-Text Injection

 Our code is based on FiD:
- Distilling Knowledge from Reader to Retriever:https://arxiv.org/abs/2012.04584. 
- [Github link to FiD](https://github.com/facebookresearch/FiD)

## Dependencies

- Python 3
- [PyTorch](http://pytorch.org/) (>= 1.6.0)
- [Transformers](http://huggingface.co/transformers/) (**version 3.0.2**)
- [NumPy](http://www.numpy.org/)



### Train

**Note**: 
- you can open the `.sh` file for <a href="#Parameter">parameter</a> modification.

```shell
bash run_okvqa_train.sh
```
or try full training process to get the Attention signal for iterative training

```shell
bash run_okvqa_full.sh
```


```

### Test

```shell
bash run_okvqa_tst.sh
```


