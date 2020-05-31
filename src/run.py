import FitPolys
import TestData
import pandas as pd
import sys
import random

dataset_path = "./training_csvs/scaled_dataset.csv"
df = pd.read_csv(dataset_path)
# Ideas to try to improve results:
# Normalise poly values, eg take the log of a in ax^2 + bx+ c
# Remove some of the polys which match less closely, like upper lip or upper eyebrow
# Take the difference from the neutral expression

# If no params, draw 5 random faces
if len(sys.argv) == 1:
    for i in range(0, 5):
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
    if(len(sys.argv) == 4):
        if not sys.argv[3].isdigit():
            print("The last parameter should be iteration number. eg run.py train dataset_name iterations")
            exit()
        if not sys.argv[2].endswith(".csv"):
            file_dir = sys.argv[2] + ".csv"
        else:
            file_dir = sys.argv[2]
        for i in range(0, int(sys.argv[3])):
            TestData.train("./training_csvs/{}".format(file_dir))
    else:
        TestData.train()

else:
    print("Invalid parameter(s)")
