# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 10:06:16 2019

@author: Admin
"""

import pandas as pd

def main():
    
    df=calculer()
    df.to_csv('test.csv')
    

def lecture(path):
    tab=pd.read_csv(path)
    df=pd.DataFrame(tab)
    return df

def calculer():
    df1=lecture('titanic_test.csv')
    df2=lecture('gender_submission.csv')
    
    df1['Survived']=0
    x1=df1.shape
    x2=df2.shape
    comp=0
    
    for i in range(1,x1[0]):
        for j in range(1,x2[0]):
            if df1['PassengerId'][i]==df2['PassengerId'][j]:
                df1['Survived'][i]=df2['Survived'][j]
                comp+=1
                break
    
    return df1   


'''def seperate():
    df=lecture('titanic_train.csv')
    F1=df.where(df['Survived']==0).dropna(axis=0, subset=['Survived'])
    F2=df.where(df['Survived']==1).dropna(axis=0, subset=['Survived'])
    F1.to_csv('death.csv')
    F2.to_csv('alive.csv')'''
    
