# NLP_Project

# Introduction
This repo presents 3 modules covering different techniques of abstractive Summarization. The PointerGenerator directory covers our implementation of the basic approach for Abstrative Summarization on top of which the other two modules are built(GetToThePoint and Query Based Abstractive Summarization). The latter two models, whose results we compare as a part of our analysis, are trained on the CNN dataset availalbe at https://cs.nyu.edu/%7Ekcho/DMQA/
Installation and steps to run each of the modules is explained in the sections below.

# Directory Structure
```
project
│   README.md   
│   report.pdf   
│
└───Query_based_RNN
│   │   querysum
│   │   querysum-data
│   │   GEN_SUM
│   │
│   └───GEN_SUM
│       │   attention_softmax
│       │   output_probablities
│       │   summaries
│   
└───GetToThePoint_Pretrained
|   │   test_output
|   |
│   └───test_output/test_output
│       │   articles
│       │   pointer-gen
│       │   pointer-gen-cov
│       │   reference
|
└───PointerGenerator
|   │   clean_test.py
|   │   files2rogue/
|   │   OpenNMT-py/
|   │   data/
|   |
│   └───OpenNMT-py/
│   |   │   preprocess.py
│   |   │   train.py
│   |   │   translate.py
|   |   |   requirements.txt
|   |      
│   └───data/gigaword
│       │   Giga
│       │   train
└───────────────────────────────────────
```

# PointerGenerator Model - Installation
```sh
$ pip3 install torch==0.4.1
$ pip install -U git+https://github.com/pltrdy/pyrouge
$ git clone https://github.com/priyankmodiPM/NLP_Project
$ cd NLP_Project/PointerGenerator/OpenNMT-py
$ pip3 install -r requirements.txt
```

# PointerGenerator Model - Download Dataset
For training this model, we use the Gigaword dataset, which can be found here, https://drive.google.com/file/d/0B6N7tANPyVeBNmlSX19Ld2xDU1E/view 
After downloading and extractinf the files, place the extracted folder in a data/gigaword sundirectory inside your PointerGenerator directory(The final directory structure is shown in the `Directory Structure` section)

# PointerGenerator Model - Cleaning Data
Unfortunately, the Gigaword dataset does contains a special token, <unk>, inside the train and validation sets, and it also expects models to output <unk> in the predicted summaries. Moreover, <unk> is inconsistent with what is used in the test set — UNK.

We run some simple bash scripts to replace <unk> with UNK. Make sure you are inside the PointerGenerator directory.
```sh
$ sed 's/<unk>/UNK/g' data/gigaword/train/train.article.txt > data/gigaword/train/train.article.cleaned.txt
$ sed 's/<unk>/UNK/g' data/gigaword/train/train.title.txt > data/gigaword/train/train.title.cleaned.txt
$ sed 's/<unk>/UNK/g' data/gigaword/train/valid.article.filter.txt > data/gigaword/train/valid.article.filter.cleaned.txt
$ sed 's/<unk>/UNK/g' data/gigaword/train/valid.title.filter.txt > data/gigaword/train/valid.title.filter.cleaned.txt
```
Test dataset contains some lines which only have the token -- UNK. We remove these by running a simple python script
```sh
$ clean_test.py
```

# Literature
Abstractive Summarization :
  - Abigail See, Peter J. Liu and Christopher D. Manning. Get To The Point: Summarization with Pointer-Generator Networks. (implemented for generic summarisation)
  - Xinyu Hua, Lu Wang. A Pilot Study of Domain Adaptation Effect for Neural Abstractive Summarization
  - Angela Fan, David Grangier, Michael Auli. Controllable Abstractive Summarization
  - Linqing Liu, Yao Lu, Min Yang, Qiang Qu, Jia Zhu, Hongyan Li. Generative Adversarial Network for Abstractive Text Summarization

Query Focused Summarisation :
  - Tal Baumel, Matan Eyal, Michael Elhadad. Query Focused Abstractive Summarization: Incorporating Query Relevance, Multi-Document Coverage, and Summary Length Constraints into seq2seq Models
  - (Paper used for paper presentation) Johan Hasselqvist, Niklas Helmertz, Mikael Kågebäck. Query-Based Abstractive Summarization Using Neural Networks. Link to the paper (implemented for query focused summarisation)
