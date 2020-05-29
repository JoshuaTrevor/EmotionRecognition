import FitPolys
import pandas as pd

df_orig = pd.read_csv("./dataset.csv")

# Draw the first 5 faces with polynomials overlayed
for i in range(0, 5):
    FitPolys.draw_face(df_orig, i)