# Twitter bot detction using Graph Neural Networks and NLP Architectures

Author: Tsingelis Konstantinos

Supervisor: Prof. Askoynis Dimitris (NTUA)

## Introduction

The task of bot account detection is crucial and demanding. Existing methods can generally be categorized into two groups: methods based on feature extraction and methods using deep learning networks. The former extracts user features from tweets and account information and feeds them into traditional machine learning classifiers, while the latter relies on deep neural network architectures. Despite initial positive results, finding a model that efficiently addresses the challenges of the issue and generalizes to the real Twitter sphere remains an open question. In the proposed model we utilize multi-modal information for each user without relying solely on feature engineering.We combine supervised and unsupervised machine learning techniques to categorize users into bots or genuine users. Specifically, we apply Natural Language Processing (NLP) models to extract information from unstructured data (tweets) and neural networks to find representations of user features, selecting those that optimize our model. Subsequently, we construct a heterogeneous graph that covers the following relationships that develop on Twitter: follower and following. We apply Graph Neural Networks (GNNs) to include social activity in our predictions. Finally, based on the integrated Twibot-20 dataset that serves as a reference point, we conduct experiments that highlight the efficiency of our model and its competitive performance compared to existing implementations.


## Dataset

More details at TwiBot-20 data , please download 'Twibot-20.zip' to the folder which also contains 'Dataset.py' and extract it there.

## Code Description

### Dataset.py
           
                     class Twibot20(self,root='./Data/,device='cpu',process=True,save=True)

root - the folder where the processed data is saved , the default folder is './Data' , which has already been created

save - whether to save the processed data or not (set it to True can save you a lot of time if you want to run this model for further ablation study)

process - If you have already saved the processed data,set it to True

### Model.py

Contains the code configuration for our proposed method. The overview of the proposed architecture:

![Architecture of the proposed model](https://github.com/TsingelisK/Diploma-Thesis/assets/147607129/a8cba69d-db09-4875-9dcb-5392badc19a9)

### Main.py

The main code that combines Dataset.py and Model.py

### Clustering.py
'
The code for configurating the optimal k for the kMeans algorithm.
