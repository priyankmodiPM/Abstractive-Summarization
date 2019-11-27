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

# PointerGenerator Model - Data Cleaning 
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
# PointerGenerator Model - Preprocessing the Data
We need to let OpenNMT-py scan the raw texts, build a vocabulary, tokenize and truncate the texts if necessary, and finally save the results to the disk.

Pick a shard_size(The higher the requested size is, the more accurate the results will be, but also, the more expensive it will be to compute the final results that works with your local memory.) That mainly affects the training process. I’ve found that it does not affect the preprocessing process, as larger datasets can quickly make the system out of memory regardless of the value of shared_size.

```sh
python3 OpenNMT-py/preprocess.py \
 -train_src data/gigaword/train/train.article.cleaned.txt \
 -train_tgt data/gigaword/train/train.title.cleaned.txt \
 -valid_src data/gigaword/train/valid.article.filter.cleaned.txt \
 -valid_tgt data/gigaword/train/valid.title.filter.cleaned.txt \
 -save_data data/gigaword/PREPROCESSED \
 -src_seq_length 10000 \
 -dynamic_dict \
 -share_vocab \
 -shard_size 200000
```
# PointerGenerator Model - Training the Model
Execute the following command to train the model. I've used the paramters as mentioned below. We set the decay rate very high to converge quickly as we had limited time and resources. 
```sh
python3 -u OpenNMT-py/train.py \
 -save_model data/gigaword/models_v2/ \
 -data data/gigaword/PREPROCESSED \
 -copy_attn \
 -global_attention mlp \
 -word_vec_size 128 \
 -rnn_size 512 \
 -layers 2 \
 -encoder_type brnn \
 -train_steps 2000000 \
 -report_every 100 \
 -valid_steps 10000 \
 -valid_batch_size 32 \
 -max_generator_batches 128 \
 -save_checkpoint_steps 10000 \
 -max_grad_norm 2 \
 -dropout 0.1 \
 -batch_size 16 \
 -optim adagrad \
 -learning_rate 0.15 \
 -start_decay_steps 100000 \
 -decay_steps 50000 \
 -adagrad_accumulator_init 0.1 \
 -reuse_copy_attn \
 -copy_loss_by_seqlength \
 -bridge \
 -seed 919 \
 -gpu_ranks 0 \
 -log_file train.v2.log
 ```

# PointerGenerator Model - Summarize Test Documents
Finally, we run our trained model on the test documents. The arguments can be understood in detail from the [offical documentation] 
```sh
python OpenNMT-py/translate.py -gpu 1 \
 -batch_size 1 \
 -beam_size 5 \
 -model data/gigaword/models_v2/_step_480000.pt \
 -src data/gigaword/Giga/input_cleaned.txt \
 -share_vocab \
 -output data/gigaword/Giga/test.pred \
 -min_length 6 \
 -verbose \
 -stepwise_penalty \
 -coverage_penalty summary \
 -beta 5 \
 -length_penalty wu \
 -alpha 0.9 \
 -block_ngram_repeat 3 \
 -ignore_when_blocking "." \
 -replace_unk
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

   [official documentation]: <http://opennmt.net/OpenNMT-py/options/translate.html>
