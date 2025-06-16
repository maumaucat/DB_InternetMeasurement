from collections import Counter
import geopandas as gpd
import numpy as np
import pandas as pd

NETWORKTYPES = ['kein Netz', '2G', '3G', '4G', '5G']
NETWORKTYPES_TRANS = ['no network', '2G', '3G', '4G', '5G']

def calculate_most_common_network_per_geometry(grid, column):
    """Calculate the most common value per geometry."""
    if column not in grid.columns:
        print(f"Column {column} not found in grid")
        raise ValueError(f"Column {column} not found in grid. Please provide a valid column name.")

    summary = grid.groupby('geometry')[column].agg(lambda x: Counter(x).most_common(1)[0][0]).apply(lambda x: NETWORKTYPES_TRANS[NETWORKTYPES.index(x)]).reset_index()
    return gpd.GeoDataFrame(summary, geometry='geometry', crs=grid.crs)

def calculate_all_per_geometry(grid, column):
    """Calculate all values per geometry."""
    if column not in grid.columns:
        print(f"Column {column} not found in grid")
        raise ValueError(f"Column {column} not found in grid. Please provide a valid column name.")
    grouped = grid.groupby('geometry')[column].agg(lambda x: ', '.join(sorted(set(str(v) for v in x if pd.notna(v))))).reset_index()

    return gpd.GeoDataFrame(grouped, geometry='geometry', crs=grid.crs)

def calculate_worst_network_per_geometry(grid, column):
    """Calculate the worst value per geometry."""
    if column not in grid.columns:
        print(f"Column {column} not found in grid")
        raise ValueError(f"Column {column} not found in grid. Please provide a valid column name.")

    # add indx column for NETWORKTYPES
    summary = grid.groupby('geometry')[column].agg(lambda x: min(x.apply(NETWORKTYPES.index))).apply(lambda x: NETWORKTYPES_TRANS[x]).reset_index()
    return gpd.GeoDataFrame(summary, geometry='geometry', crs=grid.crs)

def count_rows_per_geometry(grid, count_column):
    """Count rows per geometry and return a GeoDataFrame."""
    if count_column not in grid.columns:
        print(f"Column {count_column} not found in grid. Using default 'Messpunkt'.")
        raise ValueError(f"Column {count_column} not found in grid. Please provide a valid column name.")

    # count rows per geometry
    num_points = grid.groupby('geometry').agg({count_column: 'count'})
    # Reset index to have 'geometry' as a column instead of index
    num_points = num_points.reset_index()
    # convert to GeoDataFrame
    num_points = gpd.GeoDataFrame(num_points, geometry='geometry', crs=grid.crs)
    return num_points

def calculate_avg_value_per_geometry(grid, column):
    """Calculate the average value per geometry."""
    if column not in grid.columns:
        print(f"Column {column} not found in grid")
        raise ValueError(f"Column {column} not found in grid. Please provide a valid column name.")

    # make sure column is numeric
    grid[column] = pd.to_numeric(grid[column], errors='coerce')

    summary = grid.groupby('geometry')[column].mean().reset_index()
    return gpd.GeoDataFrame(summary, geometry='geometry', crs=grid.crs)

def calculate_median_value_per_geometry(grid, column):
    """Calculate the median value per geometry."""
    if column not in grid.columns:
        print(f"Column {column} not found in grid")
        raise ValueError(f"Column {column} not found in grid. Please provide a valid column name.")

    # make sure column is numeric
    grid[column] = pd.to_numeric(grid[column], errors='coerce')

    summary = grid.groupby('geometry')[column].median().reset_index()
    return gpd.GeoDataFrame(summary, geometry='geometry', crs=grid.crs)

def calculate_median_value_per_geometry_multiple_columns(grid, columns):
    """Calculate the median value per geometry for multiple columns."""
    for column in columns:
        if column not in grid.columns:
            print(f"Column {column} not found in grid")
            raise ValueError(f"Column {column} not found in grid. Please provide a valid column name.")

    # make sure columns are numeric
    grid[columns] = grid[columns].apply(pd.to_numeric, errors='coerce')
    # create a new column that contains all values from the specified columns # and remove NaN values
    grid['all_columns'] = grid[columns].apply(lambda x: [v for v in x if pd.notna(v)], axis=1)
    # group by geometry and aggregate the lists of values
    grouped = grid.groupby('geometry')['all_columns'].agg(lambda lists: [v for sublist in lists for v in sublist])
    # calculate the median for each geometry
    grouped_median = grouped.apply(lambda lst: np.median(lst) if lst else np.nan)
    # convert the result back to a DataFrame
    grouped_median = grouped_median.reset_index().rename(columns={'all_columns': 'median'})
    # create a GeoDataFrame
    summary = gpd.GeoDataFrame(grouped_median, geometry='geometry', crs=grid.crs)
    return summary

def calculate_avg_value_per_geometry_multiple_columns(grid, columns):
    """Calculate the average value per geometry for multiple columns."""
    for column in columns:
        if column not in grid.columns:
            print(f"Column {column} not found in grid")
            raise ValueError(f"Column {column} not found in grid. Please provide a valid column name.")

    # make sure columns are numeric
    grid[columns] = grid[columns].apply(pd.to_numeric, errors='coerce')

    summary = grid.groupby('geometry')[columns].mean().reset_index()
    # calculate mean over all columns
    summary['mean'] = summary[columns].mean(axis=1)
    return gpd.GeoDataFrame(summary, geometry='geometry', crs=grid.crs)

def percentage_of_empty_per_geometry(grid, columns):
    """Calculate the percentage of empty values per geometry."""
    for column in columns:
        if column not in grid.columns:
            print(f"Column {column} not found in grid")
            raise ValueError(f"Column {column} not found in grid. Please provide a valid column name.")

    # get the pings that are an empty
    empty_pings = grid[columns].isna().all(axis=1)

    # count rows per geometry
    num_empty_pings = grid[empty_pings].groupby('geometry').size().reset_index(name='empty_count')
    num_total_pings = grid.groupby('geometry').size().reset_index(name='total_count')

    # merge the two dataframes
    merged = num_empty_pings.merge(num_total_pings, on='geometry', how='outer').fillna(0)
    merged['percentage_empty'] = (merged['empty_count'] / merged['total_count']) * 100

    # convert to GeoDataFrame
    return gpd.GeoDataFrame(merged, geometry='geometry', crs=grid.crs)

def calculate_avg_value_per_networktype(grid, column):
    """Calculate the average value per network type."""
    if column not in grid.columns:
        print(f"Column {column} not found in grid")
        raise ValueError(f"Column {column} not found in grid. Please provide a valid column name.")

    # make sure column is numeric
    grid[column] = pd.to_numeric(grid[column], errors='coerce')

    # group by network type and geometry, then calculate the mean
    grouped = grid.groupby(['geometry', 'Typ'])[column].mean().reset_index()
    return gpd.GeoDataFrame(grouped, geometry='geometry', crs=grid.crs)

def calculate_percentage_overall(grid, column):
    """Calculate the percentage of each value in a column."""
    if column not in grid.columns:
        print(f"Column {column} not found in grid")
        raise ValueError(f"Column {column} not found in grid. Please provide a valid column name.")

    # count occurrences of each value
    counts = grid[column].value_counts(normalize=True) * 100
    counts = counts.reset_index()
    counts.columns = [column, 'percentage']

    return counts

def calculate_average_latency_overall(grid, columns):
    """Calculate the average latency overall."""
    for column in columns:
        if column not in grid.columns:
            print(f"Column {column} not found in grid")
            raise ValueError(f"Column {column} not found in grid. Please provide a valid column name.")

    # make sure columns are numeric
    grid[columns] = grid[columns].apply(pd.to_numeric, errors='coerce')
    # calculate the average latency for each column
    avg_latency = grid[columns].mean().reset_index()
    avg_latency.columns = ['column', 'average_latency']

    with open('average_latency.txt', 'w') as f:
        f.write(f'Average Latency: \n{avg_latency}\n')

    return avg_latency