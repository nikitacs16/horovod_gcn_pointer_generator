# On Incorporating Structural Information to Improve Dialogue Response Generation
[arXiv](https://arxiv.org/abs/2005.14315)

The code is based on [Q-GTTP](https://github.com/nikitacs16/q_pointer_generator) which is based on [Pointer-Generator Network](https://github.com/abisee/pointer-generator) 

We consider the task of generating dialogue responses from background knowledge comprising of domain specific resources. Specifically, given a conversation around a movie, the task is to generate the next response based on background knowledge about the movie such as the plot, review, Reddit comments etc. This requires capturing structural, sequential and semantic information from the conversation context and the background resources. This is a new task and has not received much attention from the community. We propose a new architecture that uses the ability of BERT to capture deep contextualized representations in conjunction with explicit structure and sequence information. More specifically, we use (i) Graph Convolutional Networks (GCNs) to capture structural information, (ii) LSTMs to capture sequential information and (iii) BERT for the deep contextualized representations that capture semantic information. We analyze the proposed architecture extensively. To this end, we propose a plug-and-play Semantics-Sequences-Structures (SSS) framework which allows us to effectively combine such linguistic information. Through a series of experiments we make some interesting observations. First, we observe that the popular adaptation of the GCN model for NLP tasks where structural information (GCNs) was added on top of sequential information (LSTMs) performs poorly on our task. This leads us to explore interesting ways of combining semantic and structural information to improve the performance. Second, we observe that while BERT already outperforms other deep contextualized representations such as ELMo, it still benefits from the additional structural information explicitly added using GCNs. This is a bit surprising given the recent claims that BERT already captures structural information. Lastly, the proposed SSS framework gives an improvement of 7.95% over the baseline.

![SSS Framework][logo]

[logo]: https://github.com/nikitacs16/horovod_gcn_pointer_generator/blob/master/SSSFramework.png

## Requirements
Tensorflow v1.8 and above

[Horovod](https://github.com/horovod/horovod)


## Pre-Processed Data
[Download](https://drive.google.com/open?id=1PrtMQaXwiPDHNZamVBXyFcDHKGjtnlVQ)

## To Update
1. Upload example config files
2. Add Pre-processing scrips
3. SSS Choices
4. Unused options for other experiments

## How to run (Instructions as per the original Repository)

### Run training
To train your model, run:
```
python run_summarization.py --mode=train --config_file=/path/to/config_file.yaml 
```
This will create a subdirectory of your specified `log_root` called `myexperiment` where all checkpoints and other data will be saved. Then the model will start training using the `train_*.bin` files as training data.
These fields are available *config.yaml*

### Run eval
You will have to run eval job at the end of the training to obtain model with the best 
```
python run_summarization.py --mode=eval --config_file=/path/to/config_file.yaml 
```
This saves the best model in the train directory
### Run testing
To run beam search decoding:
```
python run_summarization.py --mode=decode --config_file=/path/to/config_file.yaml
```
The beam size is specified in the *config.yaml* file.

## ToDo (Future Engineering Changes)
1. Use vector Highway connections
2. Check for GCN variations for the query. (Our initial experiments did not find infusing GCN+LSTM for query useful)
3. Explore for other tasks
4. Check other ways to combine information for Par-GCN-LSTM
