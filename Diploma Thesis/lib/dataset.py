# -*- coding: utf-8 -*-
"""Dataset

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/182Nv9m1ELRzS6mTaTrOZjdiJgI2C-ZaR
"""

import torch
import numpy as np
import pandas as pd
import json
import os
!pip install transformers
from transformers import pipeline
from datetime import datetime as dt
from torch.utils.data import Dataset
from tqdm import tqdm


class Twibot20(Dataset):
    def __init__(self,root='./Data/',device='cpu',process=True,save=True):
        self.root = root
        self.device = device
        self.process=process
        if process:
            print('Loading train.json')
            df_train=pd.read_json('./Twibot-20/train.json')
            print('Loading test.json')
            df_test=pd.read_json('./Twibot-20/test.json')
            print('Loading support.json')
            df_support=pd.read_json('./Twibot-20/support.json')
            print('Loading dev.json')
            df_dev=pd.read_json('./Twibot-20/dev.json')
            print('Finished')
            df_train=df_train.iloc[:,[0,1,2,3,5]]
            df_test=df_test.iloc[:,[0,1,2,3,5]]
            df_support=df_support.iloc[:,[0,1,2,3]]
            df_dev=df_dev.iloc[:,[0,1,2,3,5]]
            df_support['label']='None'
            self.df_data_labeled=pd.concat([df_train,df_dev,df_test],ignore_index=True)
            self.df_data=pd.concat([df_train,df_dev,df_test,df_support],ignore_index=True)
            self.df_data=self.df_data
            self.df_data_labeled=self.df_data_labeled
            self.save=save

    def load_labels(self):
        print('Loading labels...',end='   ')
        path=self.root+'label.pt'
        if not os.path.exists(path):
            labels=torch.LongTensor(self.df_data_labeled['label']).to(self.device)
            if self.save:
                torch.save(labels,'./Data/label.pt')
        else:
            labels=torch.load(self.root+"label.pt").to(self.device)
        print('Finished')

        return labels

    def Des_Preprocess(self):
        print('Loading raw feature1...',end='   ')
        path=self.root+'description.npy'
        if not os.path.exists(path):
            description=[]
            for i in range (self.df_data.shape[0]):
                if self.df_data['profile'][i] is None or self.df_data['profile'][i]['description'] is None:
                    description.append('None')
                else:
                    description.append(self.df_data['profile'][i]['description'])
            description=np.array(description)
            if self.save:
                np.save(path,description)
        else:
            description=np.load(path,allow_pickle=True)
        print('Finished')
        return description

    def Des_embbeding(self):
        print('Running feature1 embedding')
        path=self.root+"des_tensor.pt"
        if not os.path.exists(path):
            description=np.load(self.root+'description.npy',allow_pickle=True)
            print('Loading RoBerta')
            feature_extraction = pipeline('feature-extraction', model="distilroberta-base", tokenizer="distilroberta-base",device=0)
            des_vec=[]
            #for (j,each) in tqdm(enumerate(description)):
            for each in tqdm(description):
                feature=torch.Tensor(feature_extraction(each))
                for (i,tensor) in enumerate(feature[0]):
                    if i==0:
                        feature_tensor=tensor
                    else:
                        feature_tensor+=tensor
                feature_tensor/=feature.shape[1]
                des_vec.append(feature_tensor)
                #if (j%1000==0):
                    #print('[{:>6d}/229580]'.format(j+1))
            des_tensor=torch.stack(des_vec,0).to(self.device)
            if self.save:
                torch.save(des_tensor,'./Data/des_tensor.pt')
        else:
            des_tensor=torch.load(self.root+"des_tensor.pt").to(self.device)
        print('Finished')
        return des_tensor

    def tweets_preprocess(self):
        print('Loading raw feature2...',end='   ')
        path=self.root+'tweets.npy'
        if not os.path.exists(path):
            tweets=[]
            for i in range (self.df_data.shape[0]):
                one_usr_tweets=[]
                if self.df_data['tweet'][i] is None:
                    one_usr_tweets.append('')
                else:
                    for each in self.df_data['tweet'][i]:
                        one_usr_tweets.append(each)
                tweets.append(one_usr_tweets)
            tweets=np.array(tweets)
            if self.save:
                np.save(path,tweets)
        else:
            tweets=np.load(path,allow_pickle=True)
        print('Finished')
        return tweets



    def tweets_embedding(self):
        print('Running feature2 embedding')

        tweets=np.load("./Data/tweets.npy",allow_pickle=True)

        print('Initialize kMeans algorith for k=5 (based on the clustering analysis we performed in clustering.py)')
        cls = KMeans(n_clusters=5).fit(tweets.astype('double'))
        counter={}
        total={}
        for i in range(5):
            counter['cluster'+str(i)]=0
            total['cluster'+str(i)]=0

        print('Loading RoBerta')
        feature_extract=pipeline('feature-extraction',model='roberta-base',tokenizer='roberta-base',device=0,padding=True, truncation=True,max_length=500, add_special_tokens = True)
        tweets_list=[]
        for each_person_tweets in tqdm(tweets):
            for j,each_tweet in enumerate(each_person_tweets):
                each_tweet_tensor=torch.tensor(feature_extract(each_tweet))
                for k,each_word_tensor in enumerate(each_tweet_tensor[0]):
                    if k==0:
                       total_word_tensor=each_word_tensor
                    else:
                        total_word_tensor+=each_word_tensor
                total_word_tensor/=each_tweet_tensor.shape[1]
            x=np.reshape(total_word_tensor,(1,-1))
            x=x.astype('double')
            cluster=cls.predict(x)
            counter['cluster'+str(cluster[0])]+=1
            total['cluster'+str(cluster[0])]+=total_word_tensor

        for i in range(5):
            if counter['cluster'+str(i)]==0:
               usr_embed_after_clustering['cluster'+str(i)].append(torch.zeros(768))
            else:
               usr_embed_after_clustering['cluster'+str(i)].append(total['cluster'+str(i)]/counter['cluster'+str(i)])

        return usr_embed_after_clustering['cluster0'],usr_embed_after_clustering['cluster1'],usr_embed_after_clustering['cluster2'],usr_embed_after_clustering['cluster3'],usr_embed_after_clustering['cluster4']



    def num_prop_preprocess(self):
        print('Processing feature3...',end='   ')
        path0=self.root+'num_properties_tensor.pt'
        if not os.path.exists(path0):
            path=self.root
            if not os.path.exists(path+"followers_count.pt"):
                followers_count=[]
                for i in range (self.df_data.shape[0]):
                    if self.df_data['profile'][i] is None or self.df_data['profile'][i]['followers_count'] is None:
                        followers_count.append(0)
                    else:
                        followers_count.append(self.df_data['profile'][i]['followers_count'])
                followers_count=torch.tensor(np.array(followers_count,dtype=np.float32)).to(self.device)
                if self.save:
                    torch.save(followers_count,path+"followers_count.pt")

                friends_count=[]
                for i in range (self.df_data.shape[0]):
                    if self.df_data['profile'][i] is None or self.df_data['profile'][i]['friends_count'] is None:
                        friends_count.append(0)
                    else:
                        friends_count.append(self.df_data['profile'][i]['friends_count'])
                friends_count=torch.tensor(np.array(friends_count,dtype=np.float32)).to(self.device)
                if self.save:
                    torch.save(friends_count,path+'friends_count.pt')

                screen_name_length=[]
                for i in range (self.df_data.shape[0]):
                    if self.df_data['profile'][i] is None or self.df_data['profile'][i]['screen_name'] is None:
                        screen_name_length.append(0)
                    else:
                        screen_name_length.append(len(self.df_data['profile'][i]['screen_name']))
                screen_name_length=torch.tensor(np.array(screen_name_length,dtype=np.float32)).to(self.device)
                if self.save:
                    torch.save(screen_name_length,path+'screen_name_length.pt')

                favourites_count=[]
                for i in range (self.df_data.shape[0]):
                    if self.df_data['profile'][i] is None or self.df_data['profile'][i]['favourites_count'] is None:
                        favourites_count.append(0)
                    else:
                        favourites_count.append(self.df_data['profile'][i]['favourites_count'])
                favourites_count=torch.tensor(np.array(favourites_count,dtype=np.float32)).to(self.device)
                if self.save:
                    torch.save(favourites_count,path+'favourites_count.pt')

                active_days=[]
                date0=dt.strptime('Tue Sep 1 00:00:00 +0000 2020 ','%a %b %d %X %z %Y ')
                for i in range (self.df_data.shape[0]):
                    if self.df_data['profile'][i] is None or self.df_data['profile'][i]['created_at'] is None:
                        active_days.append(0)
                    else:
                        date=dt.strptime(self.df_data['profile'][i]['created_at'],'%a %b %d %X %z %Y ')
                        active_days.append((date0-date).days)
                active_days=torch.tensor(np.array(active_days,dtype=np.float32)).to(self.device)
                if self.save:
                    torch.save(active_days,path+'active_days.pt')

                statuses_count=[]
                for i in range (self.df_data.shape[0]):
                    if self.df_data['profile'][i] is None or self.df_data['profile'][i]['statuses_count'] is None:
                        statuses_count.append(0)
                    else:
                        statuses_count.append(int(self.df_data['profile'][i]['statuses_count']))
                statuses_count=torch.tensor(np.array(statuses_count,dtype=np.float32)).to(self.device)
                if self.save:
                    torch.save(statuses_count,path+'statuses_count.pt')

            else:
                active_days=torch.load(path+"active_days.pt")
                screen_name_length=torch.load(path+"screen_name_length.pt")
                favourites_count=torch.load(path+"favourites_count.pt")
                followers_count=torch.load(path+"followers_count.pt")
                friends_count=torch.load(path+"friends_count.pt")
                statuses_count=torch.load(path+"statuses_count.pt")

            active_days=pd.Series(active_days.to('cpu').detach().numpy())
            active_days=(active_days-active_days.mean())/active_days.std()
            active_days=torch.tensor(np.array(active_days))

            screen_name_length=pd.Series(screen_name_length.to('cpu').detach().numpy())
            screen_name_length_days=(screen_name_length-screen_name_length.mean())/screen_name_length.std()
            screen_name_length_days=torch.tensor(np.array(screen_name_length_days))

            favourites_count=pd.Series(favourites_count.to('cpu').detach().numpy())
            favourites_count=(favourites_count-favourites_count.mean())/favourites_count.std()
            favourites_count=torch.tensor(np.array(favourites_count))

            followers_count=pd.Series(followers_count.to('cpu').detach().numpy())
            followers_count=(followers_count-followers_count.mean())/followers_count.std()
            followers_count=torch.tensor(np.array(followers_count))

            friends_count=pd.Series(friends_count.to('cpu').detach().numpy())
            friends_count=(friends_count-friends_count.mean())/friends_count.std()
            friends_count=torch.tensor(np.array(friends_count))

            statuses_count=pd.Series(statuses_count.to('cpu').detach().numpy())
            statuses_count=(statuses_count-statuses_count.mean())/statuses_count.std()
            statuses_count=torch.tensor(np.array(statuses_count))

            num_prop=torch.cat((followers_count.reshape([229580,1]),friends_count.reshape([229580,1]),favourites_count.reshape([229580,1]),statuses_count.reshape([229580,1]),screen_name_length_days.reshape([229580,1]),active_days.reshape([229580,1])),1).to(self.device)

            if self.save:
                torch.save(num_prop,"./Data/num_properties_tensor.pt")

        else:
            num_prop=torch.load(self.root+"num_properties_tensor.pt").to(self.device)
        print('Finished')
        return num_prop

    def cat_prop_preprocess(self):
        print('Processing feature4...',end='   ')
        path=self.root+'cat_properties_tensor.pt'
        if not os.path.exists(path):
            category_properties=[]
            properties=['protected','geo_enabled','verified','contributors_enabled','is_translator','is_translation_enabled','profile_background_tile','profile_use_background_image','has_extended_profile','default_profile','default_profile_image']
            for i in range (self.df_data.shape[0]):
                prop=[]
                if self.df_data['profile'][i] is None:
                    for i in range(11):
                        prop.append(0)
                else:
                    for each in properties:
                        if self.df_data['profile'][i][each] is None:
                            prop.append(0)
                        else:
                            if self.df_data['profile'][i][each] == "True ":
                                prop.append(1)
                            else:
                                prop.append(0)
                prop=np.array(prop)
                category_properties.append(prop)
            category_properties=torch.tensor(np.array(category_properties,dtype=np.float32)).to(self.device)
            if self.save:
                torch.save(category_properties,self.root+'cat_properties_tensor.pt')
        else:
            category_properties=torch.load(self.root+"cat_properties_tensor.pt").to(self.device)
        print('Finished')
        return category_properties

    def Build_Graph(self):
        print('Building graph',end='   ')
        path=self.root+'edge_index.pt'
        if not os.path.exists(path):
            id2index_dict={id:index for index,id in enumerate(self.df_data['ID'])}
            edge_index=[]
            edge_type=[]
            for i,relation in enumerate(self.df_data['neighbor']):
                if relation is not None:
                    for each_id in relation['following']:
                        try:
                            target_id=id2index_dict[int(each_id)]
                        except KeyError:
                            continue
                        else:
                            edge_index.append([i,target_id])
                        edge_type.append(0)
                    for each_id in relation['follower']:
                        try:
                            target_id=id2index_dict[int(each_id)]
                        except KeyError:
                            continue
                        else:
                            edge_index.append([i,target_id])
                        edge_type.append(1)
                else:
                    continue
            edge_index=torch.tensor(edge_index,dtype=torch.long).t().contiguous().to(self.device)
            edge_type=torch.tensor(edge_type,dtype=torch.long).to(self.device)
            if self.save:
                torch.save(edge_index,self.root+"edge_index.pt")
                torch.save(edge_type,self.root+"edge_type.pt")
        else:
            edge_index=torch.load(self.root+"edge_index.pt").to(self.device)
            edge_type=torch.load(self.root+"edge_type.pt").to(self.device)
            print('Finished')
        return edge_index,edge_type

    def train_val_test_mask(self):
        train_idx=range(8278)
        val_idx=range(8278,8278+2365)
        test_idx=range(8278+2365,8278+2365+1183)
        return train_idx,val_idx,test_idx

    def dataloader(self):
        labels=self.load_labels()
        if self.process:
            self.Des_Preprocess()
            self.tweets_preprocess()

        des_tensor=self.Des_embbeding()
        usr_embed_after_clustering['cluster0'],usr_embed_after_clustering['cluster1'],usr_embed_after_clustering['cluster2'],usr_embed_after_clustering['cluster3'],usr_embed_after_clustering['cluster4']=self.tweets_embedding()
        num_prop=self.num_prop_preprocess()
        category_prop=self.cat_prop_preprocess()
        edge_index,edge_type=self.Build_Graph()
        train_idx,val_idx,test_idx=self.train_val_test_mask()
        return des_tensor,usr_embed_after_clustering['cluster0'],usr_embed_after_clustering['cluster1'],usr_embed_after_clustering['cluster2'],usr_embed_after_clustering['cluster3'],usr_embed_after_clustering['cluster4'],num_prop,category_prop,edge_index,edge_type,labels,train_idx,val_idx,test_idx