from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import seaborn as sns
import pandas as pd

class ModelBuilder:
  def __init__(self):
    self.names = [
    "Nearest Neighbors",
    "Linear SVM",
    "Decision Tree",
    "Logistic_Regression"]

    self.classifiers = [
    KNeighborsClassifier(3),
    SVC(kernel="linear", C=0.025),
    DecisionTreeClassifier(max_depth=5),
    LogisticRegression()]


  def split_test_train(self, features,labels,size,state):
    X_train,X_test,y_train,y_test = train_test_split(features,
                                                labels,
                                                test_size=size,
                                                random_state=state)
    return (X_train,X_test,y_train,y_test)

  def choose_best_model(self,X_test,X_train,y_test,y_train):
    scores = []
    best_model = self.classifiers[0]
    best_score = 0
    for name, clf in zip(self.names,self.classifiers):
      clf.fit(X_train,y_train)
      score = clf.score(X_test,y_test)
      scores.append(score)
      if score > best_score:
        best_model = clf
        best_score = score
    model_scores_df = pd.DataFrame()
    model_scores_df['Algorithm'] = self.names
    model_scores_df['Score'] = scores

    cm = sns.light_palette('orange',as_cmap=True)
    model_scores = model_scores_df.style.background_gradient(cmap=cm)
    print(f"The model that will perform well with {best_model}")
    return model_scores, best_model