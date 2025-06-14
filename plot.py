from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shapely
import matplotlib.patches as mpatches
import contextily as cx
from collections import Counter
import matplotlib.colors as mcolors


DATAROOT = Path('data')
OPERATORS = ['Vodafone', 'Telekom', 'o2']


def create_grid(gdf=None, bounds=None, n_cells=10, overlap=False, crs="EPSG:29902"):
    """Create square grid that covers a geodataframe area
    or a fixed boundary with x-y coords
    returns: a GeoDataFrame of grid polygons
    see https://james-brennan.github.io/posts/fast_gridding_geopandas/
    """

    if bounds is not None:
        xmin, ymin, xmax, ymax= bounds
    else:
        xmin, ymin, xmax, ymax= gdf.total_bounds

    # get cell size
    cell_size = (xmax-xmin)/n_cells
    # create the cells in a loop
    grid_cells = []
    for x0 in np.arange(xmin, xmax+cell_size, cell_size ):
        for y0 in np.arange(ymin, ymax+cell_size, cell_size):
            x1 = x0-cell_size
            y1 = y0+cell_size
            poly = shapely.geometry.box(x0, y0, x1, y1)
            grid_cells.append( poly )

    cells = gpd.GeoDataFrame(grid_cells, columns=['geometry'],
                                     crs=crs)
    if overlap:
        cells = cells.sjoin(gdf, how='inner')
    return cells

def plot_most_common_values(grid, column, map_source=cx.providers.OpenStreetMap.Mapnik, color_mapping = None ,result_path=None, title=""):
    """Plot the most common values in a gridcell of a GeoDataFrame."""

    # Gruppieren nach Geometrie und häufigsten Wert bestimmen
    def most_common(x):
        return Counter(x).most_common(1)[0][0]

    summary = grid.groupby('geometry')[column].agg(most_common).reset_index()
    summary = gpd.GeoDataFrame(summary, geometry='geometry', crs=grid.crs)

    if color_mapping is None:
        cmap = plt.get_cmap('tab10')
        color_mapping = {val: cmap(i % 10) for i, val in enumerate(summary[column].unique())}

    summary['color'] = summary[column].map(color_mapping)
    fig, ax = plt.subplots(figsize=(16, 9))
    summary.plot(color=summary['color'], edgecolor='black', linewidth=0.2, ax=ax)

    patches = [mpatches.Patch(color=color, label=label) for label, color in color_mapping.items()]
    ax.legend(handles=patches, title=column, loc='center left', bbox_to_anchor=(1, 0.5))
    ax.set_title(title)

    fig.subplots_adjust(right=0.8)  # Platz rechts schaffen

    cx.add_basemap(ax, crs=summary.crs, source=map_source)
    ax.set_axis_off()

    # Optional speichern
    if result_path:
        plt.savefig(result_path, dpi=500, bbox_inches='tight')
    else:
        plt.show()

def plot_number_of_data_points(grid, map_source=cx.providers.OpenStreetMap.Mapnik, result_path=None, count_column='Messpunkt', title=""):
    """Plot the number of data points in each grid cell."""
    # count rows per geometry
    num_points = grid.groupby('geometry').agg({count_column: 'count'})
    # Reset index to have 'geometry' as a column instead of index
    num_points = num_points.reset_index()
    # convert to GeoDataFrame
    num_points = gpd.GeoDataFrame(num_points, geometry='geometry', crs=gdf.crs)
    # Plot the number of data points

    ax = num_points.plot(column=count_column, cmap='viridis', legend=True, figsize=(16, 9))
    ax.set_title(title)
    # Add basemap
    cx.add_basemap(ax, crs=gdf.crs, source=map_source)
    # Show the plot
    if result_path:
        plt.savefig(result_path, dpi=500, bbox_inches='tight')
    else:
        plt.show()

def plot_avg_value(grid, column, map_source=cx.providers.OpenStreetMap.Mapnik, result_path=None, title="", scale_bounds=None):
    """Plot the average value in a grid cell of a GeoDataFrame."""
    cols = column if isinstance(column, list) else [column]
    for col in cols:
        if col in grid.columns:
            grid[col] = pd.to_numeric(grid[col], errors='coerce')
        else:
            print(f"Column {col} not found in grid. Skipping.")

    # Gruppieren und Mittelwerte berechnen
    grouped = grid.groupby('geometry')[cols].mean().reset_index()
    grouped = gpd.GeoDataFrame(grouped, geometry='geometry', crs=grid.crs)

    # Falls mehrere Spalten: Mittelwert über die Spalten berechnen für Plot
    if len(cols) > 1:
        grouped['avg_value'] = grouped[cols].mean(axis=1)
        plot_column = 'avg_value'
    else:
        plot_column = cols[0]

    # Plot erstellen
    fig, ax = plt.subplots(figsize=(16, 9))

    # Normierung vorbereiten (wenn gewünscht)
    if scale_bounds:
        vmin, vmax = scale_bounds
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        grouped.plot(column=plot_column, cmap='viridis', legend=True, ax=ax, norm=norm)
    else:
        grouped.plot(column=plot_column, cmap='viridis', legend=True, ax=ax)


    ax.set_title(title)

    # Basemap hinzufügen
    cx.add_basemap(ax, crs=grouped.crs, source=map_source)
    ax.set_axis_off()

    # Ergebnis speichern oder anzeigen
    if result_path:
        plt.savefig(result_path, dpi=500, bbox_inches='tight')
    else:
        plt.show()

def plot_no_pings(grid, column, map_source=cx.providers.OpenStreetMap.Mapnik, result_path=None, title="", scale_bounds=None):
    """Plot the number of data points in each grid cell."""
    cols = column if isinstance(column, list) else [column]
    for col in cols:
        if col in grid.columns:
            grid[col] = pd.to_numeric(grid[col], errors='coerce')
        else:
            print(f"Column {col} not found in grid. Skipping.")
    # get the pings that are an empty
    empty_pings = grid[cols].isna().all(axis=1)

    # count rows per geometry
    num_empty_pings = grid[empty_pings].groupby('geometry').size().reset_index(name='empty_count')
    # convert to GeoDataFrame
    num_empty_pings = gpd.GeoDataFrame(num_empty_pings, geometry='geometry', crs=grid.crs)
    # Plot the number of data points
    ax = num_empty_pings.plot(column='empty_count', cmap='viridis', legend=True, figsize=(16, 9))
    ax.set_title(title)
    # Add basemap
    cx.add_basemap(ax, crs=grid.crs, source=map_source)
    # Show the plot
    if result_path:
        plt.savefig(result_path, dpi=500, bbox_inches='tight')
    else:
        plt.show()

def plot_percentage_no_ping(grid, column, map_source=cx.providers.OpenStreetMap.Mapnik, result_path=None, title="", scale_bounds=None):
    """Plot the percentage of empty pings in each grid cell."""
    cols = column if isinstance(column, list) else [column]
    for col in cols:
        if col in grid.columns:
            grid[col] = pd.to_numeric(grid[col], errors='coerce')
        else:
            print(f"Column {col} not found in grid. Skipping.")

    # get the pings that are an empty
    empty_pings = grid[cols].isna().all(axis=1)

    # count rows per geometry
    num_empty_pings = grid[empty_pings].groupby('geometry').size().reset_index(name='empty_count')
    num_total_pings = grid.groupby('geometry').size().reset_index(name='total_count')

    # merge the two dataframes
    merged = num_empty_pings.merge(num_total_pings, on='geometry', how='outer').fillna(0)
    merged['percentage_empty'] = (merged['empty_count'] / merged['total_count']) * 100

    # convert to GeoDataFrame
    merged = gpd.GeoDataFrame(merged, geometry='geometry', crs=grid.crs)

    # Plot the percentage of empty pings
    ax = merged.plot(column='percentage_empty', cmap='viridis', legend=True, figsize=(16, 9))
    ax.set_title(title)
    # Add basemap
    cx.add_basemap(ax, crs=grid.crs, source=map_source)
    # Show the plot
    if result_path:
        plt.savefig(result_path, dpi=500, bbox_inches='tight')
    else:
        plt.show()


def load_network_availability_data(file_path):
    """Load network availability data from a CSV file."""
    network_availability = {}

    for operator in OPERATORS:
        # All Netzverfügbarkeit files for the operator
        dfs = []
        files = DATAROOT.glob(f"*/{operator}/Netzverfügbarkeit_*.csv")
        for file in files:
            df = pd.read_csv(file)
            dfs.append(df)

        if dfs:
            # Concatenate all dataframes for the operator
            concat_pd = pd.concat(dfs, ignore_index=True)
            points = gpd.points_from_xy(concat_pd['Längengrad'], concat_pd['Breitengrad'])
            gdf = gpd.GeoDataFrame(concat_pd, geometry=points, crs={'init': 'epsg:4326'})
            network_availability[operator] = gdf

        else:
            print(f"No data found for operator {operator}")

def load_network_latency_data(file_path):
    """Load network latency data from a CSV file."""
    network_latency = {}
    for operator in OPERATORS:
        dfs = []
        files = DATAROOT.glob(f"*/{operator}/netzlog_adaptive_*.csv")
        for file in files:
            df = pd.read_csv(file)
            dfs.append(df)
        if dfs:
            # Concatenate all dataframes for the operator
            concat_pd = pd.concat(dfs, ignore_index=True)
            points = gpd.points_from_xy(concat_pd['longitude'], concat_pd['latitude'])
            gdf = gpd.GeoDataFrame(concat_pd, geometry=points, crs={'init': 'epsg:4326'})
            network_latency[operator] = gdf
    return network_latency

# Evaluate data from Netzverfügbarkeit

network_availability = load_network_availability_data(DATAROOT)
network_latency = load_network_latency_data(DATAROOT)
print("Evaluating network availability data...")
rssi_min = np.inf
rssi_max = -np.inf
rsrp_min = np.inf
rsrp_max = -np.inf
rsrq_min = np.inf
rsrq_max = -np.inf

for operator, gdf in network_availability.items():

    gdf['Signalstärke (RSSI) [dBm]'] = pd.to_numeric(gdf['Signalstärke (RSSI) [dBm]'], errors='coerce')
    gdf['Signalstärke (RSRP) [dBm]'] = pd.to_numeric(gdf['Signalstärke (RSRP) [dBm]'], errors='coerce')
    gdf['Signalqualität (RSRQ) [dBm]'] = pd.to_numeric(gdf['Signalqualität (RSRQ) [dBm]'], errors='coerce')

    # Update min and max values for RSSI, RSRP and RSRQ
    rssi_min = min(rssi_min, gdf['Signalstärke (RSSI) [dBm]'].min())
    rssi_max = max(rssi_max, gdf['Signalstärke (RSSI) [dBm]'].max())
    rsrp_min = min(rsrp_min, gdf['Signalstärke (RSRP) [dBm]'].min())
    rsrp_max = max(rsrp_max, gdf['Signalstärke (RSRP) [dBm]'].max())
    rsrq_min = min(rsrq_min, gdf['Signalqualität (RSRQ) [dBm]'].min())
    rsrq_max = max(rsrq_max, gdf['Signalqualität (RSRQ) [dBm]'].max())


for operator, gdf in network_availability.items():

    print(f"Operator: {operator}, Number of data points: {len(gdf)}")
    grid = create_grid(gdf=gdf, n_cells=200, overlap=True, crs=gdf.crs)

    # Plot the number of data points in each grid cell
    plot_number_of_data_points(grid, result_path=f"results/{operator}/num_points_availability", title="Number of Data Points in Grid Cells")
    # Plot the most common network availability value in each grid cell
    plot_most_common_values(grid, column='Typ', result_path=f"results/{operator}/most_common_Typ", color_mapping={'5G': 'green', '4G': 'yellow', '3G': 'orange', '2G': 'red', 'kein Netz': 'black'}, title="Most Common Network Type in Grid Cells")
    # Plot the average network RSSI value in each grid cell
    plot_avg_value(grid, column='Signalstärke (RSSI) [dBm]', result_path=f"results/{operator}/avg_RSSI", title="Average RSSI in Grid Cells", scale_bounds=(rssi_min, rssi_max))
    # Plot the average network RSRP value in each grid cell
    plot_avg_value(grid, column='Signalstärke (RSRP) [dBm]', result_path=f"results/{operator}/avg_RSRP", title="Average RSRP in Grid Cells", scale_bounds=(rsrp_min, rsrp_max))
    # Plot the average network RSRQ value in each grid cell
    plot_avg_value(grid, column='Signalqualität (RSRQ) [dBm]', result_path=f"results/{operator}/avg_RSRQ", title="Average RSRQ in Grid Cells", scale_bounds=(rsrq_min, rsrq_max))
# Evaluate data from netzlog_adaptive
print("Evaluating network latency data...")

# Initialize min and max values for latency
latency_min = np.inf
latency_max = -np.inf

for operator, gdf in network_latency.items():
    gdf['ping_8.8.8.8'] = pd.to_numeric(gdf['ping_8.8.8.8'], errors='coerce')
    gdf['ping_1.1.1.1'] = pd.to_numeric(gdf['ping_1.1.1.1'], errors='coerce')
    gdf['ping_9.9.9.9'] = pd.to_numeric(gdf['ping_9.9.9.9'], errors='coerce')

    # Update min and max values for latency
    latency_min = min(latency_min, gdf[['ping_8.8.8.8', 'ping_1.1.1.1', 'ping_9.9.9.9']].min().min())
    latency_max = max(latency_max, gdf[['ping_8.8.8.8', 'ping_1.1.1.1', 'ping_9.9.9.9']].max().max())



for operator, gdf in network_latency.items():
    print(f"Operator: {operator}, Number of data points: {len(gdf)}")
    grid = create_grid(gdf=gdf, n_cells=200, overlap=True, crs=gdf.crs)

    plot_percentage_no_ping(grid, column=['ping_8.8.8.8', 'ping_1.1.1.1', 'ping_9.9.9.9'], title=f"Percentage of Empty Pings in Grid Cells ({operator})")
    plot_no_pings(grid, column=['ping_8.8.8.8', 'ping_1.1.1.1', 'ping_9.9.9.9'], title=f"Number of Empty Pings in Grid Cells ({operator})")
    # Plot the number of data points in each grid cell

    plot_number_of_data_points(grid, count_column='timestamp', result_path=f"results/{operator}/num_points_latency", title="Number of Data Points in Grid Cells")
    # Plot the average network latency in each grid cell
    plot_avg_value(grid, column='ping_8.8.8.8', result_path=f"results/{operator}/avg_latency_8_8_8_8", title="Average Latency to ping_8.8.8.8 in Grid Cells in ms")

    plot_avg_value(grid, column=['ping_1.1.1.1', 'ping_8.8.8.8', 'ping_9.9.9.9'], result_path=f"results/{operator}/avg_latency_overall", title="Average Latency in Grid Cells in ms", scale_bounds=(latency_min, latency_max))



