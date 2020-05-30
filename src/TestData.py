import numpy as np
from numpy.random import randint
import pandas as pd
from IPython.display import display
from sklearn.metrics import r2_score
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

#dataset = pd.read_csv("dataset.csv", sep = ",")
dataset = pd.read_csv("output_scaled.csv", sep = ",")
dimensions = dataset[["height", "width", "left", "top"]]

# for col in dataset:
#     if col.endswith("x"):
#         dataset[col] = ((dataset[col]-dataset["left"]) / dataset["width"])
#     if col.endswith("y"):
#         dataset[col] = ((dataset[col]-dataset["top"]) / dataset["height"])
        
dataset.to_csv("scaled_dataset.csv")

dataset = dataset.drop(["expression_id", "height", "width", "left", "top"], axis = 1)

# for n, val in enumerate(["Anger", "Contempt", "Disgust", "Fear", "Happy", "Sad", "Surprise"], start = 1):
#     dataset.truth_value = dataset.truth_value.replace(n, val)

dataset.replace([np.inf, -np.inf], np.nan)
dataset.fillna(0)

print(str(type(dataset.truth_value[0])))
# print(str(type(dataset["Unnamed: 0.1"][0])))
print(str(dataset.shape))
dataset

features = dataset.drop(["id", "truth_value"], axis = 1)
features = dataset.drop(["Unnamed: 0", "Unnamed: 0.1", "truth_value"], axis = 1)
value = dataset[["truth_value"]]

train_x, test_x, train_y, test_y = train_test_split(features, value, test_size = 0.20, random_state = 42)

best_crit = ""
best_depth = 0
best_min_sam_split = 0
best_max_feat = 0
best_score = 0
for crit in ["gini", "entropy"]:
    for depth in range(5, 50):
        for min_sam in [2]:
            for max_feat in range(25, 75):
                dtc = DecisionTreeClassifier(criterion = crit, max_depth = depth, min_samples_split = min_sam, max_features = max_feat, random_state = 69)
                model = dtc.fit(train_x, train_y)
                predictions = model.predict(test_x)
                score = accuracy_score(test_y, predictions)
                if score > best_score:
                    best_score = score
                    best_crit = crit
                    best_depth = depth
                    best_min_sam_split = min_sam
                    best_max_feat = max_feat
                    print("Best Score: " + str(best_score))
                    print("Best Crit: " + str(best_crit))
                    print("Best Depth: " + str(best_depth))
                    print("Best Min Sample Split: " + str(best_min_sam_split))
                    print("Best Max Features: " + str(best_max_feat) + "\n")
print("Best Score: " + str(best_score))
print("Best Crit: " + str(best_crit))
print("Best Depth: " + str(best_depth))
print("Best Min Sample Split: " + str(best_min_sam_split))
print("Best Max Features: " + str(best_max_feat) + "\n")