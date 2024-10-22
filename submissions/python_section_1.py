from typing import Dict, List
import re
import polyline
from typing import List, Tuple
import numpy as np

import pandas as pd


def reverse_by_n_elements(lst: List[int], n: int) -> List[int]:
    """
    Reverses the input list by groups of n elements.
    """
    # Your code goes here.
    result = []
    
    for i in range(0, len(lst), n):

        chunk = lst[i:i+n]
        
        for j in range(len(chunk) // 2):

            chunk[j], chunk[len(chunk) - 1 - j] = chunk[len(chunk) - 1 - j], chunk[j]
        
        result.extend(chunk)
    
    return result



def group_by_length(lst: List[str]) -> Dict[int, List[str]]:
 
    """
    Groups the strings by their length and returns a dictionary.
    """
    # Your code here
    length_dict = {}
    
    for string in lst:

        length = len(string)  
        
        if length not in length_dict:
            length_dict[length] = []
        
        length_dict[length].append(string)
    
    return dict(sorted(length_dict.items()))



def flatten_dict(nested_dict: Dict, sep: str = '.') -> Dict:

    """
    Flattens a nested dictionary into a single-level dictionary with dot notation for keys.
    
    :param nested_dict: The dictionary object to flatten
    :param sep: The separator to use between parent and child keys (defaults to '.')
    :return: A flattened dictionary
    """
    flattened = {}
    
    def flatten(current_dict: Dict[str], parent_key: str = ''):
        for key, value in current_dict.items():
            new_key = f"{parent_key}{sep}{key}" if parent_key else key
            
            if isinstance(value, dict):  
                flatten(value, new_key)
            elif isinstance(value, list):  
                for index, item in enumerate(value):
                    list_key = f"{new_key}[{index}]"
                    if isinstance(item, dict):  
                        flatten(item, list_key)
                    else:
                        flattened[list_key] = item  
            else:  
                flattened[new_key] = value

    flatten(nested_dict)
    return flattened


def unique_permutations(nums: List[int]) -> List[List[int]]:
    """
    Generate all unique permutations of a list that may contain duplicates.
    
    :param nums: List of integers (may contain duplicates)
    :return: List of unique permutations
    """
    def backtrack(path, used):

        if len(path) == len(nums):
            result.append(path[:])  
            return

        for i in range(len(nums)):
            if used[i]:
                continue
            if i > 0 and nums[i] == nums[i - 1] and not used[i - 1]:
                continue
            
            used[i] = True
            path.append(nums[i])
            backtrack(path, used)  
            path.pop()  
            used[i] = False  

    result = []
    nums.sort()  
    used = [False] * len(nums)  
    backtrack([], used)  
    return result


def find_all_dates(text: str) -> List[str]:
    """
    This function takes a string as input and returns a list of valid dates
    in 'dd-mm-yyyy', 'mm/dd/yyyy', or 'yyyy.mm.dd' format found in the string.
    
    Parameters:
    text (str): A string containing the dates in various formats.

    Returns:
    List[str]: A list of valid dates in the formats specified.
    """
    patterns = [
        r'\b(\d{2})-(\d{2})-(\d{4})\b',  
        r'\b(\d{2})/(\d{2})/(\d{4})\b', 
        r'\b(\d{4})\.(\d{2})\.(\d{2})\b'  
    ]
    
    combined_pattern = '|'.join(patterns)
    
    matches = re.findall(combined_pattern, text)
    
    valid_dates = []
    for match in matches:

        if match[0]:  
            valid_dates.append(f"{match[0]}-{match[1]}-{match[2]}")
        elif match[3]:  
            valid_dates.append(f"{match[3]}/{match[4]}/{match[5]}")
        elif match[6]:  
            valid_dates.append(f"{match[6]}.{match[7]}.{match[8]}")
    
    return valid_dates


def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    
    R = 6371000  
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    delta_phi = np.radians(lat2 - lat1)
    delta_lambda = np.radians(lon2 - lon1)

    a = np.sin(delta_phi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2.0) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    return R * c

def polyline_to_dataframe(polyline_str: str) -> pd.DataFrame:
    """
    Converts a polyline string into a DataFrame with latitude, longitude, and distance between consecutive points.
    
    Args:
        polyline_str (str): The encoded polyline string.

    Returns:
        pd.DataFrame: A DataFrame containing latitude, longitude, and distance in meters.
    """
    coordinates: List[Tuple[float, float]] = polyline.decode(polyline_str)

    latitudes = []
    longitudes = []
    distances = []

    for i, (lat, lon) in enumerate(coordinates):
        latitudes.append(lat)
        longitudes.append(lon)

        if i == 0:
            distances.append(0)  
        else:

            distance = haversine(latitudes[i - 1], longitudes[i - 1], lat, lon)
            distances.append(distance)

    df = pd.DataFrame({
        'latitude': latitudes,
        'longitude': longitudes,
        'distance': distances
    })

    return df


def rotate_and_multiply_matrix(matrix: List[List[int]]) -> List[List[int]]:
    """
    Rotate the given matrix by 90 degrees clockwise, then replace each element 
    with the sum of its original row and column excluding itself.
    
    Args:
    - matrix (List[List[int]]): 2D list representing the matrix to be transformed.
    
    Returns:
    - List[List[int]]: A new 2D list representing the transformed matrix.
    """
    n = len(matrix)
    
    rotated_matrix = [[matrix[n - j - 1][i] for j in range(n)] for i in range(n)]
    
    final_matrix = [[0] * n for _ in range(n)]

    for i in range(n):
        for j in range(n):

            row_sum = sum(rotated_matrix[i]) - rotated_matrix[i][j]
            col_sum = sum(rotated_matrix[k][j] for k in range(n)) - rotated_matrix[i][j]
            final_matrix[i][j] = row_sum + col_sum
    
    return final_matrix


def time_check(df: pd.DataFrame) -> pd.Series:
    """
    Use shared dataset-2 to verify the completeness of the data by checking whether the timestamps 
    for each unique (`id`, `id_2`) pair cover a full 24-hour and 7 days period

    Args:
        df (pandas.DataFrame): The input DataFrame containing the dataset.

    Returns:
        pd.Series: A boolean series indicating the completeness of timestamps for each (id, id_2) pair.
    """
    
    df['start'] = pd.to_datetime(df['startDay'] + ' ' + df['startTime'])
    df['end'] = pd.to_datetime(df['endDay'] + ' ' + df['endTime'])
    
    grouped = df.groupby(['id', 'id_2'])
    
    results = {}
    
    for (id_value, id_2_value), group in grouped:

        unique_days = group['start'].dt.day_name().unique()
        full_day_count = len(unique_days) == 7
        
        start_time = group['start'].min().time()  # earliest start time
        end_time = group['end'].max().time()      # latest end time
        full_24_hours = (start_time <= pd.Timestamp('00:00:00').time() and
                         end_time >= pd.Timestamp('23:59:59').time())
        
        results[(id_value, id_2_value)] = full_day_count and full_24_hours
    
    return pd.Series(results)