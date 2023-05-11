# LaKo

![](https://img.shields.io/badge/version-1.0.1-blue)
[![license](https://img.shields.io/github/license/mashape/apistatus.svg?maxAge=2592000)](https://github.com/hackerchenzhuo/LaKo/blob/main/LICENSE)
[![arxiv badge](https://img.shields.io/badge/arxiv-2207.12888-red)](https://arxiv.org/abs/2207.12888)
 - [*LaKo: Knowledge-driven Visual Question Answering via Late Knowledge-to-Text Injection*](https://arxiv.org/abs/2207.12888) 
 

>In this paper, we propose LaKo, a knowledge-driven VQA method via Late Knowledge-to-text Injection. To effectively incorporate an external KG, we transfer triples into text and propose a late injection mechanism. Finally we address VQA as a text generation task with an effective encoder-decoder paradigm. 



## 🌈 Model Architecture
![Model_architecture](https://github.com/hackerchenzhuo/LaKo/blob/main/figure/github.png)

## 📚 Dependencies

- Python 3
- [PyTorch](http://pytorch.org/) (>= 1.6.0)
- [Transformers](http://huggingface.co/transformers/) (**version 3.0.2**)
- [NumPy](http://www.numpy.org/)
- faiss-cpu



### 🚀 Train
<img align="right" alt="GIF" src="https://github.com/hackerchenzhuo/LaKo/blob/main/figure/Decoder.gif"  width="40%" height="auto" />

```shell
bash run_okvqa_train.sh
```
or try full training process to get the Attention signal for iterative training

```shell
bash run_okvqa_full.sh
```


### 🚀 Test

```shell
bash run_okvqa_test.sh
```

### ❗ Note
- ```(Optional)``` You can first **pre-train** LaKo (large version) on ```VQA2.0``` then **re-train** on ```OKVQA``` for better performance.
- You can open the `.sh` file for <a href="#Parameter">parameter</a> modification.
- The latest Transformers (e.g., 4.XX.XX) have some differences from the older version, which may lead to some unexpected error.


### Our code is based on FiD:
- Distilling Knowledge from Reader to Retriever:https://arxiv.org/abs/2012.04584. 
- [Github link to FiD](https://github.com/facebookresearch/FiD)

## 🔬 Paradigm
<img align="middle" src="https://github.com/hackerchenzhuo/LaKo/blob/main/figure/prarad.png"  width="35%" height="auto" />

## 🤝 Cite:
Please condiser citing this paper if you use the ```code``` or ```data``` from our work.
Thanks a lot :)

```bigquery
@inproceedings{DBLP:conf/jist/0007HCGFP0Z22,
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
  booktitle = {{IJCKG}},
  pages     = {20--29},
  publisher = {{ACM}},
  year      = {2022}
}
```

<a href="https://info.flagcounter.com/VOlE"><img src="https://s11.flagcounter.com/count2/VOlE/bg_FFFFFF/txt_000000/border_F7F7F7/columns_6/maxflags_12/viewers_3/labels_0/pageviews_0/flags_0/percent_0/" alt="Flag Counter" border="0"></a>

