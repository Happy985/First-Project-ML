# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 18:12:49 2019

@author: Admin
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def lecture(path):
    tab=pd.read_csv(path)
    df=pd.DataFrame(tab)
    return df

def lecture2(S):
    df=pd.read_csv(S)
    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
    df['Embarked']=df['Embarked'].map({'Q':0, 'S':1, 'C':2})
    Data=df[['Survived' , 'Sex', 'Pclass', 'Age','SibSp' , 'Parch']]
    #methode pour remplir les cases vides de la colonne Age
    Data=pd.DataFrame(Data)
    Data.fillna(value=Data['Age'].mean(), inplace=True)
    return Data

def occurence(F,S):
    Value=[]
    G=lecture('titanic_train.csv')
    Names=list(set(G[S]))
    for i in range(len(Names)):
        Value.append(list(F[S]).count(Names[i]))
    return Names, Value

def grille():
    Data= lecture2('titanic_train.csv')
    sns_plot1=sns.pairplot(Data, hue='Survived', palette='Set1')
    sns_plot1.savefig("grille.png")
    
    sns_plot2 = sns.FacetGrid(Data, col="Pclass",  row="Sex", hue='Survived')
    sns_plot2 = sns_plot2.map(plt.hist, "Age", histtype='barstacked', stacked=True).add_legend()
    sns_plot2.savefig("PclassSex.png")

def points():
    df1=lecture('alive.csv')
    df2=lecture('death.csv')
    #G=lecture('test.csv')
    X1=list(df1['Age'])
    X2=list(df2['Age'])
    Y1=list(df1['Fare'])
    Y2=list(df2['Fare'])
    
    fig, ax = plt.subplots(1, 1)
    fig.set_size_inches(13, 6, forward=True)
    
    plt.plot(X1, Y1, 'ro', label='alive')
    plt.plot(X2, Y2, 'bo', label='death')
    plt.axis([10, 50, 1, 80])
    ax.set_xlabel('Age')
    ax.set_ylabel('Fare')
    plt.legend()
    plt.show()
    fig.savefig('PointsZoom.png')
        
def histo():
    barWidth = 0.25
    fig, ax = plt.subplots(1, 5)
    fig.set_size_inches(14, 6, forward=True)
    Type=['Pclass' , 'Sex' , 'SibSp' , 'Parch' , 'Embarked']
    df1=lecture('alive.csv')
    df2=lecture('death.csv')
    pos=151
    
    for i in range(len(Type)):

        N1,M1 =occurence(df1, Type[i])
        N2,M2 =occurence(df2, Type[i])
        r1 = np.arange(len(N1))
        r2 = [x + barWidth for x in r1]
        
        plt.subplot(pos+i)
        plt.bar(r1, M1, width=0.25, edgecolor='white', label='alive')
        plt.bar(r2, M2, width=0.25, edgecolor='white', label='death')
        plt.title(Type[i])
        plt.xticks([x + barWidth/2 for x in r1], N1)
        plt.legend()
    
    fig.savefig('Histogramme.png')


def AgePclass():
    df1=lecture('alive.csv')
    df2=lecture('death.csv')
    #G=lecture('test.csv')
    X1=list(df1['Age'])
    X2=list(df2['Age'])
    Y1=list(df1['Pclass'])
    Y2=list(df2['Pclass'])
    
    fig, ax = plt.subplots(1, 1)
    fig.set_size_inches(13, 6, forward=True)
    
    plt.plot(X1, Y1, 'ro', label='alive')
    plt.plot(X2, Y2, 'bo', label='death')
    #plt.axis([10, 80, 1, 80])
    ax.set_xlabel('Age')
    ax.set_ylabel('Pclass')
    plt.legend()
    plt.show()
    fig.savefig('AgePclass.png')