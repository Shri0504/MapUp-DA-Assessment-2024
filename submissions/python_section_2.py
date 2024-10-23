import pandas as pd
import numpy as np
from datetime import datetime, time

def calculate_distance_matrix(file_path: str) -> pd.DataFrame:
    """
    This function takes the dataset-2.csv file as input and generates a DataFrame representing the distances between IDs.
    
    Args:
        file_path (str): The path to the dataset-2.csv file.
    
    Returns:
        pd.DataFrame: A DataFrame representing the distance matrix with cumulative distances along known routes.
    """
    df = pd.read_csv("dataset-1.csv")
    
    toll_locations = pd.unique(df[['id_start', 'id_end']].values.ravel('K'))
    
    num_locations = len(toll_locations)
    distance_matrix = np.full((num_locations, num_locations), np.inf)
    
    np.fill_diagonal(distance_matrix, 0)
    
    toll_index = {toll: idx for idx, toll in enumerate(toll_locations)}
    
    for _, row in df.iterrows():
        start_idx = toll_index[row['id_start']]
        end_idx = toll_index[row['id_end']]
        distance_matrix[start_idx, end_idx] = row['distance']
        distance_matrix[end_idx, start_idx] = row['distance']  
    
    for k in range(num_locations):
        for i in range(num_locations):
            for j in range(num_locations):
                distance_matrix[i, j] = min(distance_matrix[i, j], distance_matrix[i, k] + distance_matrix[k, j])
    
    distance_df = pd.DataFrame(distance_matrix, index=toll_locations, columns=toll_locations)
    
    return distance_df




def unroll_distance_matrix(distance_df: pd.DataFrame) -> pd.DataFrame:
    """
    Unroll a distance matrix to a DataFrame in the style of the initial dataset.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame: Unrolled DataFrame containing columns 'id_start', 'id_end', and 'distance'.
    """
    data = []
    
    for id_start in distance_df.index:
        for id_end in distance_df.columns:
            if id_start != id_end:  
                distance = distance_df.loc[id_start, id_end]
                data.append({'id_start': id_start, 'id_end': id_end, 'distance': distance})
    
    unrolled_df = pd.DataFrame(data)
    
    return unrolled_df



def find_ids_within_ten_percentage_threshold(df: pd.DataFrame, reference_value: int) -> list:
    """
    Find all IDs whose average distance lies within 10% of the average distance of the reference ID.

    Args:
        df (pandas.DataFrame)
        reference_id (int)

    Returns:
        pandas.DataFrame: DataFrame with IDs whose average distance is within the specified percentage threshold
                          of the reference ID's average distance.
    """
 
    reference_distances = df[df['id_start'] == reference_value]['distance']
    
    avg_distance = reference_distances.mean()
    
    threshold_lower = avg_distance * 0.9  
    threshold_upper = avg_distance * 1.1  
    
    valid_ids = []
    
    for id_start in df['id_start'].unique():
        id_distances = df[df['id_start'] == id_start]['distance']
        id_avg_distance = id_distances.mean()
        
        if threshold_lower <= id_avg_distance <= threshold_upper:
            valid_ids.append(id_start)
    
    return sorted(valid_ids)


def calculate_toll_rate(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate toll rates for each vehicle type based on the unrolled DataFrame.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame
    """

    
    rate_coefficients = {
        'moto': 0.8,
        'car': 1.2,
        'rv': 1.5,
        'bus': 2.2,
        'truck': 3.6
    }
    
    df['moto'] = df['distance'] * rate_coefficients['moto']
    df['car'] = df['distance'] * rate_coefficients['car']
    df['rv'] = df['distance'] * rate_coefficients['rv']
    df['bus'] = df['distance'] * rate_coefficients['bus']
    df['truck'] = df['distance'] * rate_coefficients['truck']
    
    return df

def calculate_time_based_toll_rates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate time-based toll rates for different time intervals within a day.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame
    """
    
    def apply_discount(row, day, start_time, end_time):
        if day in ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']:
            if time(0, 0) <= start_time < time(10, 0):
                discount_factor = 0.8
            elif time(10, 0) <= start_time < time(18, 0):
                discount_factor = 1.2
            else:
                discount_factor = 0.8
        else:  
            discount_factor = 0.7
        
        row['moto'] *= discount_factor
        row['car'] *= discount_factor
        row['rv'] *= discount_factor
        row['bus'] *= discount_factor
        row['truck'] *= discount_factor
        
        return row
    
    time_intervals = [
        (time(0, 0), time(10, 0)),
        (time(10, 0), time(18, 0)),
        (time(18, 0), time(23, 59, 59))
    ]
    
    days_of_week = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    
    result_rows = []

    for _, row in df.iterrows():
        for day in days_of_week:
            for start_time, end_time in time_intervals:
                new_row = row.copy()
                new_row['start_day'] = day
                new_row['end_day'] = day
                new_row['start_time'] = start_time
                new_row['end_time'] = end_time
                
                new_row = apply_discount(new_row, day, start_time, end_time)
                
                result_rows.append(new_row)
    
    result_df = pd.DataFrame(result_rows)
    
    return result_df