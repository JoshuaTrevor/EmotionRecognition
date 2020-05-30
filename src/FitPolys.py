import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PolySetup import extract_poly_mappings

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
def get_colsw(substr, df):
    return [col for col in df.columns if substr in col]


def graph_poly(coefficients, x_values):
    # Calculate intermediate f(x) values to get a smooth line
    x = np.linspace(x_values[0], x_values[1], 1000)
    poly = np.polyval(coefficients, x)

    plt.plot(x, poly, 'r-')


def get_polys(feature_points):
    polys = []
    for feature in feature_points:
        x_vals = []
        y_vals = []
        for point in feature_points[feature]:
            x_vals.append(point[0])
            y_vals.append(500-point[1])
        
        if(len(x_vals) > 0 and len(x_vals) == len(y_vals)):
            polynomial_coeffs = np.polyfit(x_vals, y_vals, deg=2)
            sorted_x = np.sort(x_vals)
            lower_bound = sorted_x[0]
            upper_bound = sorted_x[-1]
            polys.append([polynomial_coeffs, [lower_bound, upper_bound]])
    return polys
        

def graph_points(df, row_index, show_overlap=False):
    x_col_list = []
    y_col_list = []
    for col in df.columns:
        if col.endswith("_x"):
            x_col_list.append(col)
        elif col.endswith("_y"):
            y_col_list.append(col)

    x_vals = []
    y_vals = []
    mapping = extract_poly_mappings()
    for x_col, y_col in zip(x_col_list, y_col_list):
        if(not show_overlap and x_col in get_replaced_points(mapping)):
            continue
        x_vals.append(df[x_col].iloc[row_index])
        y_vals.append(500-df[y_col].iloc[row_index])
    plt.scatter(x_vals, y_vals, s=2)
    

def draw_face(df, row_index, show_overlap=False):
    mapping = extract_poly_mappings()
    points = get_poly_points(mapping, df, row_index)
    graph_points(df, row_index, show_overlap)
    polys = get_polys(points)

    for poly in polys:
        graph_poly(poly[0], poly[1])

    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()


# Get a list of points which can be replaced by polygons
def get_replaced_points(mapping):
    # Flatten map values
    obselete_cols = set()
    for key in mapping:
        for value in mapping[key]:
            obselete_cols.add(value+"_x")
            obselete_cols.add(value+"_y")
    return obselete_cols


def replace_points(df):
    mapping = extract_poly_mappings()

    # Remove points which are replaced by polys
    obselete_cols = get_replaced_points(mapping)
    reduced_df = df.drop(obselete_cols, axis=1)

    poly_df = create_poly_df(df, mapping)

    return pd.concat([reduced_df, poly_df], axis=1, sort=False)
    

def create_poly_csv(df):
    poly_df = replace_points(df)
    poly_df.to_csv("output.csv")


def flatten_poly_attribs(poly):
    attribs = []
    for elem in poly:
        for sub_elem in elem:
            attribs.append(sub_elem)
    return attribs

def create_poly_df(df, mapping):
    rows = []

    for row_index in range(0, len(df.index)):
        row_poly_attribs = [] # An unordered 1d list of poly coeffs and bounds
        points = get_poly_points(mapping, df, row_index)
        polys = get_polys(points)
        for poly in polys:
            for attrib in flatten_poly_attribs(poly):
                row_poly_attribs.append(attrib)
        rows.append(row_poly_attribs)
    
    return pd.DataFrame(rows)