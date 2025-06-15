from collections import Counter
import geopandas as gpd
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
    with open('mean_columns.txt', 'w') as f:
        f.write(summary.head().to_string())
    return gpd.GeoDataFrame(summary, geometry='geometry', crs=grid.crs)