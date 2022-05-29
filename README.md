<<<<<<< HEAD


# Generalizing to the Future: Mitigating Entity Bias in Fake News Detection (ENDEF)
This is the official implementation of our paper **Generalizing to the Future: Mitigating Entity Bias in Fake News Detection**, which has been accepted by SIGIR2022. [Paper]()

The wide dissemination of fake news is increasingly threatening both individuals and society. Fake news detection aims to train a model on the past news and detect fake news of the future. Though great efforts have been made, existing fake news detection methods overlooked the unintended entity bias in the real-world data, which seriously influences models' generalization ability to future data. For example, 97% of news pieces in 2010-2017 containing the entity 'Donald Trump' are real in our data, but the percentage falls down to merely 33% in 2018. This would lead the model trained on the former set to hardly generalize to the latter, as it tends to predict news pieces about 'Donald Trump' as real for lower training loss. In this paper, we propose an entity debiasing framework (**ENDEF**) which generalizes fake news detection models to the future data by mitigating entity bias from a cause-effect perspective. Based on the causal graph among entities, news contents, and news veracity, we separately model the contribution of each cause (entities and contents) during training. In the inference stage, we remove the direct effect of the entities to mitigate entity bias. Extensive offline experiments on the English and Chinese datasets demonstrate that the proposed framework can largely improve the performance of base fake news detectors, and online tests verify its superiority in practice. To the best of our knowledge, this is the first work to explicitly improve the generalization ability of fake news detection models to the future data.

## Introduction
The proposed ENDEF is model-agnostic, and it can be implemented with diverse base models. This repository provides the implementations of ENDEF and five base models (BiGRU, EANN, BERT, MDFEND, BERT-Emo):
* BiGRU：[On the Properties of Neural Machine Translation: Encoder-Decoder Approaches](https://arxiv.org/pdf/1409.1259.pdf?ref=https://githubhelp.com)
* EANN: [EANN: Event Adversarial Neural Networks for Multi-Modal Fake News Detection](https://dl.acm.org/doi/pdf/10.1145/3219819.3219903) (KDD 2018)
* BERT: [Bert: Pre-training of Deep Bidirectional Transformers for Language Understanding](http://aclanthology.lst.uni-saarland.de/N19-1423.pdf) (NAACL 2019)
* MDFEND: [MDFEND: Multi-domain Fake News Detection](https://dl.acm.org/doi/pdf/10.1145/3459637.3482139) (CIKM 2021)
* BERT-Emo: [Mining Dual Emotion for Fake News Detection](https://dl.acm.org/doi/pdf/10.1145/3442381.3450004) (WWW 2021)

## Requirements

- Python 3.6
- PyTorch > 1.0
- Pandas
- Numpy
- Tqdm

## File Structure

```
.
├── ENDEF-SIGIR2022
│   └── readme.md
│   ENDEF_en
└── ENDEF_ch
    ├── main.py
    ├── grid_search.py
    ├── data
    │   ├── train.json
    │   ├── train_emo.npy
    │   ├── val.json
    │   ├── val_emo.npy
    │   ├── test.json
    │   └── test_emo.npy
    ├── logs
    │   ├── event
    │   ├── json   #final results
    │   ├── param
    ├── models
    │   ├── bigru.py
    │   ├── bigruendef.py
    │   └── ...
    ├── param_model   #save model parameters
    └── utils               # Ready to use
        ├── dataloader.py
        └── utils.py
```

## Run

Parameter Configuration:

- max_len: the max length of a sample, default for `170`
- early_stop: default for `5`
- epoch: training epoches, default for `50`
- aug_prob: probability of augmentation (mask and drop), default for `0.1`
- gpu: the index of gpu you will use, default for `0`
- lr: learning_rate, default for `0.0001`
- model_name: model_name within `bigru, bigru_endef, bert, bert_endef, bertemo, bertemo_endef, eann, eann_endef, mdfend, mdfend_endef`, default for `bigru_endef`

You can run this code through:

```powershell
# Reset several parameters
python main.py --gpu 1 --lr 0.0001 --model_name bigru
```

The best learning rate for various models are different: BiGRU (0.0009), EANN (0.0001), BERT (7e-05), MDFEND (7e-5), BERTEmo (7e-05).


## Reference

```
Zhu, Yongchun, et al. "Generalizing to the Future: Mitigating Entity Bias in Fake News Detection." Proceedings of the 45nd International ACM SIGIR Conference on Research and Development in Information Retrieval. 2022.
```

or in bibtex style:

=======
# ENDEF-SIGIR2022

This is the official repository of the paper:

> **Generalizing to the Future: Mitigating Entity Bias in Fake News Detection**
>
> Yongchun Zhu, Qiang Sheng, Juan Cao, Shuokai Li, Danding Wang, and Fuzhen Zhuang
>
> *Proceedings of the 45nd International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR 2022)*
> 
> Preprint: https://arxiv.org/abs/2204.09484
# Citation
>>>>>>> 0fdd91f79f542498382acbc83eb3b5d919ba11f3
```
@inproceedings{ENDEF,
    title = "Generalizing to the Future: Mitigating Entity Bias in Fake News Detection",
    author = "Zhu, Yongchun and Sheng, Qiang and Cao, Juan and Li, Shuokai and Wang, Danding and Zhuang, Fuzhen",
    booktitle = "Proceedings of the 45nd International ACM SIGIR Conference on Research and Development in Information Retrieval",
    month = July,
    year = "2022",
    publisher = "Association for Computing Machinery"
}
```