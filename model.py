import numpy as np
import pandas as pd
import re
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

def get_title(name):
    title = re.search(' ([A-Za-z]+)\.', name)
    if title:
        return title.group(1)
    return ""

def main ():
    #read data
    titanic = []
    titanic = pd.read_csv('C:/Users/ajray/Desktop/PythonTutorials/titanic/data/train.csv')

    #split between features and classes
    x = []
    y = []
    for index, row in titanic.iterrows():
        x.append(row)
        y.append(row['Survived'])

    #feature engineering
    x = pd.DataFrame(x)
    
    avg_age = x['Age'].mean()
    std_age = x['Age'].std()
    x['Age'] = x['Age'].fillna(np.random.randint(avg_age - std_age, avg_age + std_age))
    x['AgeBin'] = pd.cut(x['Age'], 5)

    x['FamilySize'] = x['SibSp'] + x['Parch'] + 1
    x['FamilySize'] = pd.cut(x['FamilySize'], 5, right=True, include_lowest=True)

    x['Title'] = x['Name'].apply(get_title)
    x['Title'] = x['Title'].replace(['Capt', 'Col', 'Countess', 'Don', 'Dr', 'Jonkheer', 'Lady', 'Major', 'Sir'], 'Unique')
    x['Title'] = x['Title'].replace(['Mlle', 'Ms', 'Mme'], 'Miss')
    #print(pd.crosstab(x['Title'], x['Sex']))
    
    x['Fare'] = pd.cut(x['Fare'],5)
    #print(pd.crosstab(x['Title'], x['Fare']))

    # print (x[["Pclass", "Survived"]].groupby(['Pclass'], as_index=False).mean())
    # print (x[["Sex", "Survived"]].groupby(['Sex'], as_index=False).mean())
    # print (x[["Embarked", "Survived"]].groupby(['Embarked'], as_index=False).mean())
    # print (x[["Title", "Survived"]].groupby(['Title'], as_index=False).mean())
    # print (x[["Fare", "Survived"]].groupby(['Fare'], as_index=False).mean())
    # print (x[["AgeBin", "Survived"]].groupby(['AgeBin'], as_index=False).mean())
    # print (x[["FamilySize", "Survived"]].groupby(['FamilySize'], as_index=False).mean())

    x = x.drop(['PassengerId', 'Survived', 'Ticket', 'Cabin', 'Age', 'SibSp', 'Parch', 'Name'], axis=1)
    x = pd.get_dummies(x, columns=['Pclass', 'Sex', 'Embarked', 'AgeBin', 'FamilySize', 'Title', 'Fare'])
    # print(x)

    #Training the Model
    x = np.array(x)
    y = np.array(y)

    X_train, X_test, y_train, y_test = train_test_split(
    x, y, test_size=0.25, random_state=0)

    # print(X_train.shape, y_train.shape)
    # print(X_test.shape, y_test.shape)

    classifiers = [
        KNeighborsClassifier(3),
        SVC(kernel="rbf", C=0.025, probability=True),
        NuSVC(probability=True),
        DecisionTreeClassifier(),
        RandomForestClassifier(),
        AdaBoostClassifier(),
        GradientBoostingClassifier(),
        GaussianNB(),
        LinearDiscriminantAnalysis(),
        QuadraticDiscriminantAnalysis()
    ]

    log_cols=["Classifier", "Accuracy", "Log Loss"]
    log = pd.DataFrame(columns=log_cols)

    for clf in classifiers:
        clf.fit(X_train, y_train)
        name = clf.__class__.__name__
        
        print("="*30)
        print(name)
        
        print('****Results****')
        train_predictions = clf.predict(X_test)
        acc = accuracy_score(y_test, train_predictions)
        print("Accuracy: {:.4%}".format(acc))
        
        train_predictions = clf.predict_proba(X_test)
        ll = log_loss(y_test, train_predictions)
        print("Log Loss: {}".format(ll))
        
        log_entry = pd.DataFrame([[name, acc*100, ll]], columns=log_cols)
        log = log.append(log_entry)
    
    print("="*30)

    sns.set_color_codes("muted")
    sns.barplot(x='Accuracy', y='Classifier', data=log, color="b")

    plt.xlabel('Accuracy %')
    plt.title('Classifier Accuracy')
    plt.show()

    sns.set_color_codes("muted")
    sns.barplot(x='Log Loss', y='Classifier', data=log, color="g")

    plt.xlabel('Log Loss')
    plt.title('Classifier Log Loss')
    plt.show()

main()