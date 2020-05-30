import numpy as np
from numpy.random import randint
import pandas as pd
from IPython.display import display
from sklearn.metrics import r2_score
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

#dataset = pd.read_csv("dataset.csv", sep = ",")
def train(dataset_path="./scaled_dataset.csv"):
    dataset = pd.read_csv(dataset_path, sep = ",")
    dimensions = dataset[["height", "width", "left", "top"]]

    dataset = dataset.drop(["expression_id", "height", "width", "left", "top"], axis = 1)

    dataset.replace([np.inf, -np.inf], np.nan)
    dataset.fillna(0)

    features = dataset.drop(["id", "truth_value"], axis = 1)
    try:
        features = dataset.drop(["Unnamed: 0", "Unnamed: 0.1", "truth_value"], axis = 1)
    except:
        print("couldn't drop")
    value = dataset[["truth_value"]]

    train_x, test_x, train_y, test_y = train_test_split(features, value, test_size = 0.20, random_state = 42)

    best_crit = ""
    best_depth = 0
    best_min_sam_split = 0
    best_max_feat = 0
    best_score = 0
    counter = 0
    print("Optimising decision tree parameters")
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
                counter+=1
                print("Iteration: {}/90 Best accuracy: {}%".format(counter, best_score*100))
    print("Best Score: " + str(best_score))
    print("Best Crit: " + str(best_crit))
    print("Best Depth: " + str(best_depth))
    print("Best Min Sample Split: " + str(best_min_sam_split))
    print("Best Max Features: " + str(best_max_feat) + "\n")
    log_result()


def log_result():
    # Save result as entry to log file
    # Store: Accuracy, date recorded, the name of the csv used and any other parameters applied to the data     
    print("log not implemented")