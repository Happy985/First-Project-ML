# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 12:18:19 2019

@author: Admin
"""

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

def courbes():
    df=pd.read_csv('titanic_train.csv')
    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
    df['Embarked']=df['Embarked'].map({'Q':0, 'S':1, 'C':2})
    
    ######a simple heatmap to see where we are missing data#####
    plt.figure(figsize=(10,6))
    sns.heatmap(df.isnull(),yticklabels=False,cbar=False,cmap='viridis')
    #####a simple hist to compare the number of survived and dead person#####
    plt.figure(figsize=(10,6))
    sns.countplot(x='Survived',data=df,palette='RdBu_r')
    
    plt.figure(figsize=(10,6))
    sns.countplot(x='Survived',data=df,hue='Sex',palette='RdBu_r')
    #####to see only one parameter by himself#####
    plt.figure(figsize=(10,6))
    sns.distplot(df['Age'].dropna(),kde=False,color='darkred',bins=30,)
    #####PlotBox#####
    plt.figure(figsize=(12, 7))
    sns.boxplot(x='Pclass',y='Age',data=df,palette='winter')
    
    