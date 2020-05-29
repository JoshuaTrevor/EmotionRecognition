import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PolySetup import extract_poly_mappings


df_orig = pd.read_csv("dataset.csv")

# Returns the coords used by each poly for a certain image
def get_poly_points(mapping, df, row_index):
    # Dict that maps a polynomial label to a set of points
    coords_dict = dict()

    for feature in mapping:
        coords_dict[feature] = []
        for point in mapping[feature]:
            x_col = point + "_x"
            y_col = point + "_y"
            x = df[x_col].iloc[row_index]
            y = df[y_col].iloc[row_index]
            pair = [x, y]
            coords_dict[feature].append(pair)
    return coords_dict



# Get columns containing a substring in their heading.
def get_colsw(substr):
    return [col for col in df_orig.columns if substr in col]


def graph_poly(coefficients, x_values):
    #Determine the range of x values the polynomial should span:
    sorted_x = np.sort(x_values)
    x = np.linspace(sorted_x[0], sorted_x[-1], 1000)
    poly = np.polyval(coefficients, x)
    plt.plot(x, poly, 'r-')

def graph_polys(feature_points):
    for feature in feature_points:
        x_vals = []
        y_vals = []
        for point in feature_points[feature]:
            x_vals.append(point[0])
            y_vals.append(500-point[1])
        
        if(len(x_vals) > 0 and len(x_vals) == len(y_vals)):
            polynomial_coeffs = np.polyfit(x_vals, y_vals, deg=2)
            graph_poly(polynomial_coeffs, x_vals)
        



def graph_points(df, row_index):
    x_col_list = []
    y_col_list = []
    for col in df.columns:
        if col.endswith("_x"):
            x_col_list.append(col)
        elif col.endswith("_y"):
            y_col_list.append(col)

    x_vals = []
    y_vals = []
    for x_col, y_col in zip(x_col_list, y_col_list):
        #print("looking at {} and {}".format(x_col, y_col))
        #print(df[x_col].iloc[0])
        x_vals.append(df[x_col].iloc[row_index])
        y_vals.append(500-df[y_col].iloc[row_index])
    plt.scatter(x_vals, y_vals)
    

def draw_face(row_index):
    mapping = extract_poly_mappings()
    points = get_poly_points(mapping, df_orig, row_index)
    graph_points(df_orig, row_index)
    graph_polys(points)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()

for i in range(0, 5):
    draw_face(i)
# print(get_colsw("mouth_upper_lip"))