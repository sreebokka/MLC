import pandas
from sklearn import tree
import pydotplus
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import matplotlib.image as pltimg
df = pandas.read_csv("machinedata.csv")
print(df)

d = {'NIL': 0, 'LOW': 1, 'MED': 2, 'HIGH':3, 'FULL':4}
df['Battery'] = df['Battery'].map(d)
d = {'YES': 1, 'NO': 0}
df['JOB'] = df['JOB'].map(d)
print(df)
features = ['Machine', 'Battery', 'MIN', 'Distance']

X = df[features]
y = df['JOB']

print(X)
print(y)
dtree = DecisionTreeClassifier()
dtree = dtree.fit(X, y)
data = tree.export_graphviz(dtree, out_file=None, feature_names=features)
graph = pydotplus.graph_from_dot_data(data)
graph.write_png('Tasktree.png')

img=pltimg.imread('Tasktree.png')
imgplot = plt.imshow(img)
plt.show()
print(dtree.predict([[4, 0, 60, 0]])) 
