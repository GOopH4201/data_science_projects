import numpy as np
import pandas as pd

def outliers_iqr(data, feature, log_scale=False, left=1.5, right=1.5, add_one=True):
    """Clearing a DataFrame of outliers using the Tukey or interquartile method

    Args:
        data (pandas.DataFrame): DataFrame with outliers
        feature (str): Column in which outliers are searched
        log_scale (bool, optional): Is the data distribution log normal?. Defaults to False.
        left (float, optional): Left brush border multiplier. Defaults to 1.5.
        right (float, optional): Right brush border multiplier. Defaults to 1.5.
        add_one (bool, optional): Add one in log. Defaults to True.

    Returns:
        pandas.DataFrame: DataFrame of outliers
        pandas.DataFrame: DataFrame without outliers
    """
    if log_scale:
        x = np.log(data[feature] + add_one)
    else:
        x = data[feature]
    Q25, Q75 = x.quantile(0.25), x.quantile(0.75)
    IQR = Q75 - Q25
    bound_lower = Q25 - left*IQR
    bound_upper = Q75 + right*IQR
    outliers = data[(x > bound_upper) | (x < bound_lower)]
    cleaned = data[(x >= bound_lower) & (x <= bound_upper)]
    return outliers, cleaned


def outliers_z_score(data, feature, auto=False, log_scale=False, left=3, right=3, add_one=True):
    """Clearing a DataFrame of outliers using the z-score method

    Args:
        data (pandas.DataFrame): DataFrame with outliers
        feature (str): Column in which outliers are searched
        auto (bool, optional): Automatically detect asymmetry. Defaults to False
        log_scale (bool, optional): Is the data distribution log normal?. Defaults to False.
        left (float, optional): Left brush border multiplier. Defaults to 3.
        right (float, optional): Right brush border multiplier. Defaults to 3.
        add_one (bool, optional): Add one in log. Defaults to True.

    Returns:
        pandas.DataFrame: DataFrame of outliers
        pandas.DataFrame: DataFrame without outliers
    """
    if log_scale:
        x = np.log(data[feature]+add_one)
    else:
        x = data[feature]
    mu = x.mean()
    sigma = x.std()
    As = x.skew() * 3
    if auto:
        if As > 0:
            left = 3 + As
        if As < 0:
            right = 3 - As
    lower_bound = mu - left * sigma
    upper_bound = mu + right * sigma
    outliers = data[(x < lower_bound) | (x > upper_bound)]
    cleaned = data[(x >= lower_bound) & (x <= upper_bound)]
    return outliers, cleaned


def find_low_inf_columns(data, ratio=0.95, ignore=None):
    """Finds uninformative columns

    Args:
        data (pandas.DataFrame): DataFrame in which the search is performed
        ratio (float, optional): Threshold percentage. Defaults to 0.95.
        ignore (list, str, optional): List of columns to ignore. Defaults to [ ].

    Returns:
        list: List of column names. Defaults to [ ].
    """
    if not ignore:
        ignore = list()
    low_information_cols = [] 

    for col in data.columns:
        if col in ignore:
            continue
        top_freq = data[col].value_counts(normalize=True).max()
        nunique_ratio = data[col].nunique() / data[col].count()
        
        if top_freq > ratio:
            low_information_cols.append(col)
        
        if nunique_ratio > ratio:
            low_information_cols.append(col)
    
    return low_information_cols


if __name__ == "__main__":
    print("This is a library for DataFrames from pandas for cleaning DF from outliers and other things.")