import FitPolys
import TestData
import pandas as pd
import sys
import os
import random

dataset_path = "./training_csvs/scaled_dataset.csv"
df = pd.read_csv(dataset_path)
# Ideas to try to improve results:
# Normalise poly values, eg take the log of a in ax^2 + bx+ c
# Remove some of the polys which match less closely, like upper lip or upper eyebrow
# Take the difference from the neutral expression

# If no params, draw 5 random faces
if len(sys.argv) == 1:
    for i in range(0, 3):
        df_length = len(df.index)
        FitPolys.draw_face(df, random.randrange(0, (df_length-1)+1), show_overlap=False)

# If parameter is a number, draw that number of random images.
elif sys.argv[1].isdigit():
    for i in range(0, int(sys.argv[1])):
        df_length = len(df.index)
        FitPolys.draw_face(df, random.randrange(0, (df_length-1)+1), show_overlap=False)

# If executed with a parameter containing "create", create a new polynomial'd csv
elif "create" in sys.argv[1].lower():
    print("Creating new csv...")
    scaled = "scaled" in dataset_path
    FitPolys.create_poly_csv(df, scaled)
    print("Done")

elif "train" in sys.argv[1].lower():
    if(len(sys.argv) == 3):
        for i in range(0, int(sys.argv[2])):
            
            for f in os.listdir("./training_csvs/"):
                print("\nTraining iteration: {} Dataset: {}".format(i, f))
                TestData.train("./training_csvs/" + f, custom_tree=False)
    else:
        TestData.train(custom_tree=False)

# Calculate the truth value distribution and performance of a random agent
elif "dist" in sys.argv[1].lower():
    ma = dict()
    for elem in df["truth_value"]:
        if elem in ma:
            ma[elem] = ma[elem] + 1
        else:
            ma[elem] = 1
    sorted_keys = sorted(ma)
    for key in sorted_keys:
        print("{} - {}".format(key, ma[key]))

    final = 0
    correct_guess = 0
    for i in range(0, len(df.index)):
        
        random_guess = random.randrange(1, 7)
        if int(round(random_guess)) == int(round(df.iloc[i]["truth_value"])):
            correct_guess += 1
        final = i
    print("{}/{} correct with random".format(correct_guess, final))
else:
    print("Invalid parameter(s)")
