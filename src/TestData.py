import numpy as np
from numpy.random import randint
import pandas as pd
from IPython.display import display
from sklearn.metrics import r2_score
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from datetime import datetime

#dataset = pd.read_csv("dataset.csv", sep = ",")
def train(dataset_path="./training_csvs/poly_output.csv"):
    dataset = pd.read_csv(dataset_path, sep = ",")
    dimensions = dataset[["height", "width", "left", "top"]]

    dataset = dataset.drop(["expression_id", "height", "width", "left", "top"], axis = 1)

    dataset.replace([np.inf, -np.inf], np.nan)
    dataset.fillna(0)

    try:
        features = dataset.drop(["id", "truth_value"], axis = 1)
    except:
        features = dataset.drop(["truth_value"], axis = 1)
    try:
        features = features.drop(["Unnamed: 0", "Unnamed: 0.1", "truth_value"], axis = 1)
    except:
        #print("couldn't drop")
        None
    value = dataset[["truth_value"]]

    train_x, test_x, train_y, test_y = train_test_split(features, value, test_size = 0.20)

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
                    dtc = DecisionTreeClassifier(criterion = crit, max_depth = depth, min_samples_split = min_sam, max_features = max_feat)
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
                # print("Iteration: {}/90 Best accuracy: {}%".format(counter, best_score*100))
    print("Best Score: " + str(best_score))
    # print("Best Crit: " + str(best_crit))
    # print("Best Depth: " + str(best_depth))
    # print("Best Min Sample Split: " + str(best_min_sam_split))
    # print("Best Max Features: " + str(best_max_feat) + "\n")
    log_result(best_score, dataset_path)


def log_result(score, dataset_path):
    acc = str(score * 100) + "%"
    dataset = dataset_path[2:]
    date = datetime.now()
    f = open("log_file.txt", "a")
    f.write("Acc: {}% Dataset: '{}' Date: {}\n".format(acc,dataset,date))
    f.close()
    update_totals()


def update_totals():
    f = open("log_file.txt", "r")
    scores = dict()
    for line in f:
        if(len(line) > 10):
            # Get the dataset of the entry
            spl = line.split("Dataset: '")[1]
            dataset = spl.split("'")[0]

            # Get the accuracy of the entry
            spl = line.split("Acc: ")[1]
            acc = float(spl.split("%")[0])

            # Associate acc with dataset
            if dataset not in scores:
                scores[dataset] = [acc]
            else:
                scores[dataset].append(acc)
    f.close()

    new_results = []
    acc_count_dict = dict()

    for dataset in scores:
        count = len(scores[dataset])
        avg_acc = np.mean(scores[dataset])
        new_results.append("Acc: {}% Attempts: {} Dataset: '{}'\n".format(avg_acc, count, dataset))
        

    f = open("results.txt", "w")
    f.writelines(new_results)
    



