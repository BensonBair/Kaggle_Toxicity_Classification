# Kaggle Competitions Toxic Classification


## Overview
In this competition, we managed to build several models which measure toxicity and identify several forms of unintended bias with respect to observed identities.

## Installation
```
pip install -r requirement  
```



## Preprocessing
We once converted all sentecnes into lower case in LSTM model, but normal case got higher LB score. 

Preprocessing steps are as follows:
 *  all http(url) are substituted into url
 *  replace all emoji symbols into ' '
 *  use flashtext to find mispelled words and replace them into original words
 *  \n\t are substituted by ' '
 *  \s{2,} are substituted by ' '

## Get statistics features
We tried to create some statistical features and added them into LSTM models during training process.

Statistical features are as follows:
* swear word 
* upper word 
* unique word
* emoji
* number of characters 


## Embedding
We used pretrained word embeddings such as:
* Fasttext
* Glove

In our experiments, fasttext is a slightly better than Glove.
We didn't concatenate fasttext and Glove embeddings due to time constraint. (Interestingly, everyone seems to train BERT models.)


## Model
### LSTM model
Our lstm models are different with public version, it consisted of lstm cells without gru cells.
* Attention didn't improve LB significantly.
* Spatial Dropout had improvement in LB score.
* Before blending these three models, the LB score of each lstm model was around 0.935x~0.938x. After blending, we got LB score of 0.93963.


### BERT model

We used pretrained bert models from: [pytorch-pretrained-BERT](https://github.com/huggingface/pytorch-pretrained-BERT) and `BertForSequenceClassification` for sequence classification.
* The results with or without text preprocessing are quite similar.

* The batch size from 16 to 32 improve the LB score. I assume that batch size effects acurracy significantly
* The learning rate we set was `2e-5`
* LB score of our single model was around 0.9415x~0.94220
* We blended five BERT models and got LB score 0.94294


### GPT2 model
* We only got LB score around 0.938 for our single GPT2 model. We decided to put more efforts on training BERT models. 

### Ensemble model

* In the end, we blended 3 LSTM models and 5 BERT models and with weights 0.3 and 0.7 respectively.
* The public LB score was 0.9443.


## References 
