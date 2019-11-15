# -*- coding: utf-8 -*-
"""
Created on Tue Nov  11 09:54:28 2019

@author: Admin
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#%matplotlib inline

def lecture(S):
    df=pd.read_csv(S)
    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
    df['Embarked']=df['Embarked'].map({'Q':0, 'S':1, 'C':2})
    Data=df[['Survived' , 'Sex', 'Pclass', 'Age']]
    #Data=df[['Survived' , 'Pclass' , 'Sex' , 'Age' , 'SibSp' , 'Parch' , 'Fare' , 'Embarked']]
    Data=pd.DataFrame(Data)
    Data.fillna(value=Data['Age'].mean(), inplace=True)
    return Data

def vectors():
    Tab1=lecture('titanic_train.csv')
    X1=Tab1.drop('Survived', axis=1)
    Y1=Tab1['Survived']
    #Test Tab
    Tab2=lecture('test.csv')
    X2=Tab2.drop('Survived', axis=1)
    Y2=Tab2['Survived']
    #Results
    return Tab1, X1, Y1, Tab2, X2, Y2 

###################### KNN methode ############################################
def bestneighbors():
    Tab1, X1, Y1, Tab2, X2, Y2 = vectors()
    error = []
    # Will take some time
    from sklearn.neighbors import KNeighborsClassifier
    for i in range(1,50):
        KNN= KNeighborsClassifier(n_neighbors=i)
        KNN.fit(X1, Y1)
        Estim=KNN.predict(X2)
        error.append(np.mean(Estim != Y2))
    # Tracer courbe afin de choisir l'erreur minimale
    plt.figure(figsize=(12,7))
    plt.plot(range(1,50), error, color='blue', linestyle='dashed', marker='o', markerfacecolor='red', markersize=10)
    plt.title('Error Rate vs. K Value')
    plt.xlabel('K')
    plt.ylabel('Error')
    return (error.index(min(error))+1)
    
def KNNClass():
    Tab1, X1, Y1, Tab2, X2, Y2 = vectors()
    n= bestneighbors()
    from sklearn.neighbors import KNeighborsClassifier
    KNN = KNeighborsClassifier(n_neighbors=n)
    KNN.fit(X1, Y1)
    Estim=KNN.predict(X2)
    from sklearn.metrics import classification_report, confusion_matrix
    print(classification_report(Y2, Estim))
    A1=confusion_matrix(Y2, Estim)
    FinalResult(Estim)
    print(A1)
###############################################################################
    
############################### KMean #########################################
def KMeanClass():
    Tab1, X1, Y1, Tab2, X2, Y2 = vectors()
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=2, n_init=20)
    kmeans.fit(X1, Y1)
    Estim=kmeans.predict(X2)
    from sklearn.metrics import classification_report,confusion_matrix
    print(classification_report(Y2,Estim))
    A1=confusion_matrix(Y2,Estim)
    print(A1)
###############################################################################
    
############################## SVM ############################################
def SVMClass():
    Tab1, X1, Y1, Tab2, X2, Y2 = vectors()
    C=[0.1,1, 10, 100]
    gamma=[1,0.1,0.01,0.001]
    v1, v2= BestSVM()
    #Forecast
    from sklearn.svm import SVC   
    MSVM=SVC(C=C[v1], gamma=gamma[v2], kernel='rbf')
    MSVM.fit(X1, Y1)
    Estim=MSVM.predict(X2)
    from sklearn.metrics import classification_report,confusion_matrix
    print(classification_report(Y2,Estim))
    A1=confusion_matrix(Y2,Estim)
    print(A1)
    #Enregistrer le vecteur d'estimation
    FinalResult(Estim)
      
def BestSVM():
    Tab1, X1, Y1, Tab2, X2, Y2 = vectors()
    C=[0.1,1, 10, 100]
    gamma=[1,0.1,0.01,0.001]
    error=[]
    #Position d'erreur minimale
    from sklearn.svm import SVC  
    for i in range(len(C)):
        for j in range(len(gamma)):
            MSVM=SVC(C=C[i], gamma=gamma[j], kernel='rbf')
            MSVM.fit(X1, Y1)
            Estim=MSVM.predict(X2)
            error.append(np.mean(Estim != Y2))
    #Tracer la courbe d'erreur    
    fig=plt.figure(figsize=(12,7))
    plt.plot(range(len(C)*len(gamma)), error, color='blue', linestyle='dashed', marker='o', markerfacecolor='red', markersize=10)
    plt.title('Error Rate vs. K Value')
    plt.xlabel('K')
    plt.ylabel('Error')
    fig.savefig('Error SVM.png')
    x=error.index(min(error))
    x1=x//len(C)
    x2=x%len(gamma)
    return x1, x2
###############################################################################  

##################### Random Forest Tree ###################################### 
def TreeClassifier():
    Tab1, X1, Y1, Tab2, X2, Y2 = vectors()
    # Preparing library to draw the final tree
    from IPython.display import Image  
    from sklearn.externals.six import StringIO  
    from sklearn.tree import export_graphviz
    import pydot 
    features = list(Tab2.columns[1:])
    print(features)
    #DecisionTreeClassifier methode
    from sklearn.tree import DecisionTreeClassifier
    dtree = DecisionTreeClassifier()
    dtree.fit(X1,Y1)
    Estim=dtree.predict(X1)
    from sklearn.metrics import classification_report,confusion_matrix
    print(classification_report(Y1,Estim))
    A1=confusion_matrix(Y1,Estim)
    print(A1)
    # Tracer l'arbre de classification
    dot_data = StringIO()  
    export_graphviz(dtree, out_file=dot_data,feature_names=features,filled=True,rounded=True)
    graph = pydot.graph_from_dot_data(dot_data.getvalue())  
    Image(graph[0].create_png()) 
    graph[0].write_pdf("Decision Tree.pdf")
    
    #RandomForestClassifier methode
    from sklearn.ensemble import RandomForestClassifier
    dtree = RandomForestClassifier(n_estimators=100)
    dtree.fit(X1,Y1)
    Estim=dtree.predict(X2)
    from sklearn.metrics import classification_report,confusion_matrix
    print(classification_report(Y2,Estim))
    A2=confusion_matrix(Y2,Estim)
    print(A2)
    FinalResult(Estim)
    #Tracer Random Forest Tree
    dot_data = StringIO()
    estimator = dtree.estimators_[5]
    export_graphviz(estimator, out_file=dot_data,feature_names=features,filled=True,rounded=True)
    graph = pydot.graph_from_dot_data(dot_data.getvalue())  
    Image(graph[0].create_png()) 
    graph[0].write_pdf("Random Forest.pdf")
    #Résultat
    return A1, A2

def comp():
    A1, A2= TreeClassifier()
    B1= np.c_[A1,A2]
    if (B1[0,1]+B1[1,0])>(B1[0,3]+B1[1,2]):
        print('RandomForestClassifier offert more True value. So it is the best classifier')
    else:
        print('DecisionTreeClassifier offert more True value. So it is the best classifier')
###############################################################################
        
def FinalResult(Estim):
    df=pd.DataFrame(Estim)
    A=pd.read_csv('test.csv')
    df['PassengerId']=A['PassengerId']
    df.rename(columns={0: 'Survived'}, inplace=True)
    df=df[['PassengerId','Survived']]
    df.set_index('PassengerId', inplace=True)
    df.to_csv('result.csv')



    
    