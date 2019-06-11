# On Incorporating Structural Information to Improve Dialogue Response Generation
Arxiv version coming soon
The code is based on [Q-GTTP](https://github.com/nikitacs16/q_pointer_generator) which is based on [Pointer-Generator Network](https://github.com/abisee/pointer-generator) 

There has been an active interest in building conversation systems that are grounded in background knowledge to improve the informativeness of a response. However, finding the relevant snippet of information within the background knowledge based on the context of the current conversation and then producing a response is indeed challenging. On closer inspection, we find that this search for incorporating the relevant information can be improved by using structural information present in the document. Graph Convolutional Networks (GCN) is one paradigm to incorporate structural information that has achieved empirical success on several NLP tasks. On empirical evaluation, we found out that the conventional adaption of GCN  is ill-suited for span-based dialogue generation (Holl-E dataset). Hence, in this work, we investigate different ways to effectively leverage structural information along with contextual information. From our experimental study, we observed that the performance depends on factors such as the combination of structural and contextual information as well as the underlying graph designed for the task. Additionally, we found that explicitly incorporating structural information also benefited deep contextualized word embeddings such as ELMo and BERT. We observe that though BERT-based architectures achieve state-of-the-art performance, simpler non-contextualized word representation-based architectures also achieve competitive performance on this task. 

## Requirements
Tensorflow v1.8 and above
[Horovod](https://github.com/horovod/horovod)
Single GPU code [here](https://github.com/nikitacs16/gcn_pointer_generator)

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
python run_summarization.py --mode=eval --config_file=/path/to/config_file.yaml 
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
