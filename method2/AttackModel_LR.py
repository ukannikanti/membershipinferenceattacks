import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load the dataset
positive_df = pd.read_csv('./method2/data/attack/train_1.csv')
negative_df = pd.read_csv('./method2/data/attack/train_0.csv')

p_list = positive_df.to_numpy().tolist()
n_list = negative_df.to_numpy().tolist()
p_list.extend(n_list)
df = pd.DataFrame(p_list)

# Clean, split the data into test & train
# Use a utility from sklearn to split and shuffle your dataset.
train_df, test_df = train_test_split(df, test_size=0.1)
train_df = np.array(train_df)
test_df = np.array(test_df)

train_labels = np.array(train_df[:, -1])
test_labels = np.array(test_df[:, -1])

train_features = np.array(train_df[:, :-1])
test_features = np.array(test_df[:, :-1])

# Normalize the data
scaler = StandardScaler()
train_features = scaler.fit_transform(train_features)
test_features = scaler.transform(test_features)
train_features = np.clip(train_features, -5, 5)
test_features = np.clip(test_features, -5, 5)

print('Training labels shape:', train_labels.shape)
print('Test labels shape:', test_labels.shape)

print('Training features shape:', train_features.shape)
print('Test features shape:', test_features.shape)

clf = LogisticRegression().fit(train_features, train_labels)
predict_proba = clf.predict_proba(test_features)
predictions = clf.predict(test_features)
print("Test Accuracy: ", accuracy_score(predictions, test_labels))

# # save
# # with open('../target_models/method_2.pkl', 'wb') as f:
# #     pickle.dump(clf, f)
#
#
#
