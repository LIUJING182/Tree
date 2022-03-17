from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OrdinalEncoder
import pandas as pd
import numpy as np
from sklearn import tree
from sklearn.model_selection import train_test_split
import pydotplus

np.set_printoptions(threshold=np.inf)
pd.set_option('display.width', 300)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

index=int(input("1. Breast Cancer Data  2. Thoracic Surgery Data \nEnter the index (1 or 2) of the data set: "))

if index!=1 and index!=2:
    print("error index")

if index==1:
    data = pd.read_csv('breast-cancer.data', header=None)
    data.iloc[:, -1] = LabelEncoder().fit_transform(data.iloc[:, -1])
    data.iloc[:, 0:-1] = OrdinalEncoder().fit_transform(data.iloc[:, 0:-1])
    data.columns = ['class', 'age', 'menoppause', 'tumor-size', 'inv-nodes', 'node-caps', 'deg-malig', 'breast', 'quad',
                    'irradiat']
    feature = data[
        ['age', 'menoppause', 'tumor-size', 'inv-nodes', 'node-caps', 'deg-malig', 'breast', 'quad', 'irradiat']]
    target = data[['class']]


else:
    data = pd.read_csv('Thoracic Surgery.csv', header =None)
    data.iloc[:,-1] = LabelEncoder().fit_transform(data.iloc[:,-1])
    data.iloc[:,0:-1] = OrdinalEncoder().fit_transform(data.iloc[:,0:-1])
    data.columns = ['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17'];
    feature = data[['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16']]
    target=data[['17']]
X_train=feature
Y_train=target
train_x, test_x, train_y, test_y = train_test_split(X_train, Y_train)
depth=int(input("Enter the depth of the tree you want to get: "))

if index==1:
    clf = tree.DecisionTreeClassifier(max_depth=depth - 1
                                      , min_samples_leaf=25
                                      , min_samples_split=25, criterion="entropy"
                                      , random_state=50
                                      , splitter="random")
else:
    clf = tree.DecisionTreeClassifier(max_depth=depth - 1,
                                      min_samples_leaf=5,
                                      min_samples_split=5,
                                      criterion="entropy",
                                      random_state=50,
                                      splitter="random")
clf.fit(train_x, train_y)
dot_data = tree.export_graphviz(clf, out_file=None,
                                    feature_names=None,
                                    class_names=["0", "1"],
                                    filled=True, rounded=True,
                                    special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data)
graph.write_png("tree.png")
print("the tree has been shown in tree.png")
