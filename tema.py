import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

train = pd.read_csv('ann-train.data',sep=" ", header=None)
cols = [22, 23]
train.drop(train.columns[cols], axis=1, inplace=True)
#print(train)
test = pd.read_csv('ann-test.data',sep=" ", header=None)
test.drop(test.columns[cols], axis=1, inplace=True)
#print(test)
test_y = test[21]
test.drop(test.columns[21], axis=1, inplace=True)

y = train[21]
train.drop(train.columns[21], axis=1, inplace=True)
x = train
#print(test_rez)

max_samples_values = [0.25, 0.5, 0.85]
max_features_values = [0.1, 0.5, 0.8]

for i in max_samples_values:
    for j in max_features_values:
        model = RandomForestClassifier(n_estimators=10, bootstrap=True, max_samples=i, max_features=j)
        model.fit(x, y)
        y_pred = model.predict(test)
        #cm = confusion_matrix(test_y, y_pred)
        #print(cm)
        #print(model.score(test, test_y))
        #print(accuracy_score(test_y, y_pred))
        print({round(accuracy_score(test_y, y_pred),4)*100})
        print(y_pred)
