import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PolySetup import extract_poly_mappings

# Returns the coords used by each poly for a certain image
def get_poly_points(mapping, df, row_index):
    # Dict that maps a polynomial label to a set of points
    coords_dict = dict()
    degrees_dict = dict()

    for feature in mapping:
        coords_dict[feature] = []
        degrees_dict[feature] = []
        for point in mapping[feature]:
            # Degrees for each feature has been added to FacePolyMappings.data
            if (point.startswith("deg:")):
                degrees_dict[feature].append(int(point[5:]))
            else:
                x_col = point + "_x"
                y_col = point + "_y"
                x = df[x_col].iloc[row_index]
                y = df[y_col].iloc[row_index]
                pair = [x, y]
                coords_dict[feature].append(pair)
    return coords_dict, degrees_dict



# Get columns containing a substring in their heading.
def get_colsw(substr, df):
    return [col for col in df.columns if substr in col]


def graph_poly(coefficients, x_values):
    # Calculate intermediate f(x) values to get a smooth line
    x = np.linspace(x_values[0], x_values[1], 1000)
    poly = np.polyval(coefficients, x)

    plt.plot(x, poly, 'r-')


#TODO Make the degree variable, defined in the mappings file. Upper lip top should be third order I think?
#DONE
def get_polys(feature_points, degrees):
    polys = []
    for feature in feature_points:
        x_vals = []
        y_vals = []
        for point in feature_points[feature]:
            x_vals.append(point[0])
            y_vals.append(500-point[1])
        
        if(len(x_vals) > 0 and len(x_vals) == len(y_vals)):
            polynomial_coeffs = np.polyfit(x_vals, y_vals, deg=degrees[feature][0])
            sorted_x = np.sort(x_vals)
            lower_bound = sorted_x[0]
            upper_bound = sorted_x[-1]
            polys.append([polynomial_coeffs, [lower_bound, upper_bound]])
    return polys
        

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
        x_vals.append(df[x_col].iloc[row_index])
        y_vals.append(500-df[y_col].iloc[row_index])
    plt.scatter(x_vals, y_vals, s=2)
    

def draw_face(df, row_index):
    mapping = extract_poly_mappings()
    points, degrees = get_poly_points(mapping, df, row_index)
    graph_points(df, row_index)
    polys = get_polys(points, degrees)

    for poly in polys:
        graph_poly(poly[0], poly[1])

    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()