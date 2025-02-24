# Functions related to the filed

import config.config_field as config_field

import numpy as np
import pandas as pd
import torch

def convert_field_grid(df, col_x, col_y):
    """
    Converts the coordinates in the specified columns of a DataFrame
    to a grid format based on the field dimensions.
    Works in place.
    Parameters:
        df: pandas.DataFrame
        col_x, col_y (str): Name of the columns containing the x and y-coordinate.
    """
   
    bin_x = np.linspace(0, config_field.FIELD_WIDTH, config_field.GRID_WIDTH + 1)
    bin_y = np.linspace(0, config_field.FIELD_HEIGHT, config_field.GRID_HEIGHT + 1)
    
    df[col_x] = df[col_x].clip(0, config_field.FIELD_WIDTH)
    df[col_y] = df[col_y].clip(0, config_field.FIELD_HEIGHT)
    
    df[col_x] = pd.cut(df[col_x], bins=bin_x, labels=False, include_lowest=True).astype(pd.Int8Dtype())
    df[col_y] = pd.cut(df[col_y], bins=bin_y, labels=False, include_lowest=True).astype(pd.Int8Dtype())

def get_centers(x_values, y_values):
    """
    Calculate the center coordinates for given x and y grid values based on field configuration.
    Parameters:
        x_values, y_values (pandas.Series or vector): x and y (grid) values.
    Returns:
        x_result, y_result (pandas.Series or vector): Center coordinates.
    """
    
    bin_x = np.linspace(0, config_field.FIELD_WIDTH, config_field.GRID_WIDTH + 1)
    bin_y = np.linspace(0, config_field.FIELD_HEIGHT, config_field.GRID_HEIGHT + 1)
    
    x_centers = (bin_x[:-1] + bin_x[1:]) / 2
    y_centers = (bin_y[:-1] + bin_y[1:]) / 2
    
    x_result = x_values.apply(lambda x: x_centers[x])
    y_result = y_values.apply(lambda y: y_centers[y])
    
    return x_result, y_result  


def get_centers_int(x_values, y_values):
    """
    Calculate the center coordinates for given x and y grid values based on field configuration.
    Parameters:
        x_values, y_values (int): x and y (grid) values.
    Returns:
        x_result, y_result (float): Center coordinates.
    """
    
    bin_x = np.linspace(0, config_field.FIELD_WIDTH, config_field.GRID_WIDTH + 1)
    bin_y = np.linspace(0, config_field.FIELD_HEIGHT, config_field.GRID_HEIGHT + 1)
    
    x_centers = (bin_x[:-1] + bin_x[1:]) / 2
    y_centers = (bin_y[:-1] + bin_y[1:]) / 2
    
    x_values = np.clip(x_values, 0, config_field.GRID_WIDTH - 1)
    y_values = np.clip(y_values, 0, config_field.GRID_HEIGHT - 1)
    
    x_result = x_centers[x_values]
    y_result = y_centers[y_values]
    
    return x_result, y_result

def get_centers_torch(x_values, y_values):
    """
    Calculate the center coordinates for given x and y grid values based on field configuration using PyTorch.
    
    Parameters:
        x_values, y_values (torch.Tensor): x and y grid values (integer grid indices).
    
    Returns:
        x_result, y_result (torch.Tensor): Center coordinates.
    """
    
    bin_x = torch.linspace(0, config_field.FIELD_WIDTH, config_field.GRID_WIDTH + 1,
                           dtype = x_values.dtype, device = x_values.device)
    bin_y = torch.linspace(0, config_field.FIELD_HEIGHT, config_field.GRID_HEIGHT + 1,
                           dtype = y_values.dtype, device = y_values.device)
    
    x_centers = (bin_x[:-1] + bin_x[1:]) / 2
    y_centers = (bin_y[:-1] + bin_y[1:]) / 2
    
    x_result = x_centers[x_values.long()]
    y_result = y_centers[y_values.long()]
    
    return x_result, y_result


def carry_line(x1, y1, x2, y2):
    """
    Generates a list of sector representing a line from (x1, y1) to (x2, y2)
    using Bresenham's line algorithm. Transform a carry to a path of movements.
    Parameters:
        x1, y1, x2, y2 (int): The coordinates of the starting-ending point.
    Returns:
        list of tuple: A list of (x, y) tuples.
    """
    
    path = []
    
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    
    sx = 1 if x1 < x2 else -1
    sy = 1 if y1 < y2 else -1
    
    err = dx - dy
    
    while True:
        
        path.append((x1, y1))
        
        if x1 == x2 and y1 == y2:
            break
        
        e2 = 2 * err
        
        mx, my = 0, 0
        
        if e2 > -dy:
            err -= dy
            x1 += sx
            mx = sx
        
        if e2 < dx:
            err += dx
            y1 += sy
            my = sy
    
    return path
