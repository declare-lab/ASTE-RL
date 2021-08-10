# Aspect Sentiment Triplet Extraction using Reinforcement Learning

**Abstract**: Aspect Sentiment Triplet Extraction (ASTE) is the task of extracting
triplets of aspect terms, their associated sentiments, and the opinion
terms that provide evidence for the expressed sentiments. Previous
approaches to ASTE usually simultaneously extract all three components or first identify the aspect and opinion terms, then pair them
up to predict their sentiment polarities. In this work, we present
a novel paradigm, ASTE-RL, by regarding the aspect and opinion
terms as arguments of the expressed sentiment in a hierarchical
reinforcement learning (RL) framework. We first focus on sentiments expressed in a sentence, then identify the target aspect and
opinion terms for that sentiment. This takes into account the mutual interactions among the triplet’s components while improving
exploration and sample efficiency. Furthermore, this hierarchical RL
setup enables us to deal with multiple and overlapping triplets. In
our experiments, we evaluate our model on existing datasets from
laptop and restaurant domains and show that it achieves state-ofthe-art performance.

🔥 :fire: 🔥 [Download the paper](https://declare-lab.net/assets/pdfs/ASTE_RL_CIKM_2021_FINAL.pdf)

## Data
### ASTE-Data-V2
ASTE-Data-V2 is originally released by the paper "Position-Aware Tagging for Aspect Sentiment Triplet Extraction". It can be downloaded here: https://github.com/xuuuluuu/SemEval-Triplet-data/tree/master/ASTE-Data-V2-EMNLP2020.


## Requirements
- torch
- numpy
- spacy
- transformers (https://huggingface.co/transformers/installation.html)
- tokenizations (https://github.com/explosion/tokenizations)


## Run
Command

```
python main.py {--[option1] [value1] --[option2] [value2] ... }
```

Change the corresponding options to set hyper-parameters:

```python
parser.add_argument('--lr', type=float, default=0.00002, help="Learning rate")
parser.add_argument('--epochPRE', type=int, default=40, help="Number of epoch on pretraining")
parser.add_argument('--epochRL', type=int, default=15, help="Number of epoch on training with RL")
parser.add_argument('--dim', type=int, default=300, help="Dimension of hidden layer")
parser.add_argument('--statedim', type=int, default=300, help="Dimension of state")
parser.add_argument('--batchsize', type=int, default=16, help="Batch size on training")
parser.add_argument('--batchsize_test', type=int, default=64, help="Batch size on testing")
parser.add_argument('--print_per_batch', type=int, default=50, help="Print results every XXX batches")
parser.add_argument('--sampleround', type=int, default=5, help="Sample round in RL")
parser.add_argument('--numprocess', type=int, default=1, help="Number of process")
parser.add_argument('--start', type=str, default='', help="Directory to load model")
parser.add_argument('--test', type=bool, default=False, help="Set to True to inference")
parser.add_argument('--pretrain', type=bool, default=False, help="Set to True to pretrain")
parser.add_argument('--datapath', type=str, default='./data/ASTE-Data-V2-EMNLP2020/14lap/', help="Data directory")
parser.add_argument('--testfile', type=str, default='test_triplets.txt', help="Filename of test file")
parser.add_argument('--dropout', type=float, default=0.5, help="Dropout")
parser.add_argument('--seed', type=int, default=1, help="PyTorch seed value")
```

Start with pretraining:
```
python main.py --datapath ./data/ASTE-Data-V2-EMNLP2020/14lap/ --pretrain True
```

Then reinforcement learning fine-tuning:
```
python main.py --lr 0.000005 --datapath ./data/ASTE-Data-V2-EMNLP2020/14lap/ --start checkpoints/{experiment_id}/model
```

Inference (results will be printed, can be modified to be saved to a file in `TrainProcess.py`):
```
python main.py --datapath ./data/ASTE-Data-V2-EMNLP2020/14lap/ --start checkpoints/{experiment_id}/model --test True --testfile test_triplets.txt
```


## Custom Data
1. Edit `Parser.py` to load the necessary files.
2. Add your data loader in `DataManager.py`.
3. Modify your data loader in `DataManager.py` to output these information:
    - **self.all_pos_tags: ['B-PRON', 'I-PRON', 'B-VERB', 'I-VERB', ...]** (_order depends on your dataset_)
    - **self.sentiments: ['POS', 'NEG', 'NEU']** (_order depends on your dataset_)
    - **self.data: {'train': [...], 'dev': [...], 'test': [...]}** (_key names depend on you_)
4. Modify your data loader in `DataManager.py` to output self.data with these information:
    - **'sentext': "I like your dog ."**
    - **'triplets': {'sentpol': 1, 'aspect': 'your dog', 'opinion': 'like', 'aspect_tags': [0,0,2,1,0], 'opinion_tags': [0,2,0,0,0]}** (_'sentpol' refers to sentiment polarity and 1 is the index+1 of the positive sentiment in self.sentiments, +1 is to account for the lack of sentiment having an index of 0. For aspect/opinion tags, 0 is the non-aspect/opinion span, 1 is the inside of the span and 2 is the beginning of the span._)
    - **'pos_tags': [0,2,0,6,16]**
    - **'bert_to_whitespace': [[0],[1],[2],[3],[4]]** (_This is used for printing inference results in their original format without BERT's subword tokenisation. It shows the alignment between the BERT tokens and whitespace tokens. In this example, since BERT's tokeniser splits the tokens into the same ones as a simple whitespace tokeniser without subwords, there is a one-to-one match between the BERT and whitespace tokens._)
    - **'whitespace_tokens': ['I', 'like', 'your', 'dog', '.']** (_This is used for printing inference results in their original format without BERT's subword tokenisation._)
5. Edit `main.py` to use your data loader.


## Acknowledgements
Our code is adapted from the code from the paper "A Hierarchical Framework for Relation Extraction with Reinforcement Learning" at https://github.com/truthless11/HRL-RE. We would like to thank the authors for their well-organised and efficient code.

## Citation
Samson Yu Bai Jian, Tapas Nayak, Navonil Majumder, Soujanya Poria. 2021. Aspect Sentiment Triplet Extraction Using Reinforcement Learning. In CIKM ’21: Proceedings of the 30th ACM International Conference on Information & Knowledge Management, November 01–05, 2021, Gold Coast, Queensland, Australia. ACM, New York, NY, USA, 5 pages.
