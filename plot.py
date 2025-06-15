from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shapely
import matplotlib.patches as mpatches
import contextily as cx
import matplotlib.colors as mcolors
from Aggregation import *

DATAROOT = Path('data')
OPERATORS = ['Vodafone', 'Telekom', 'o2']
TRANSLATIONS = { 'Typ' : 'Type'}
PINGS = ['ping_8.8.8.8', 'ping_1.1.1.1', 'ping_9.9.9.9', 'ping_google.com', 'ping_uni-osnabrueck.de', 'ping_utwente.nl']


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

def plot_categorical_values(grids, bounds, column, map_source=cx.providers.OpenStreetMap.Mapnik, result_path=None, title="", plot=True, agg_func=calculate_most_common_network_per_geometry):
    """Plot categorical values in a grid cell of a GeoDataFrame."""

    aggregated_per_operator = {}
    unique_values = set()
    for operator, grid in grids.items():
        aggregated = agg_func(grid, column)
        aggregated_per_operator[operator] = aggregated
        unique_values.update(aggregated[column].unique())

    cmap_base = plt.get_cmap('tab10')
    cmap_dict = {val: cmap_base(i % 10) for i, val in enumerate(sorted(unique_values))}

    # Create a figure and axis
    fig, axes = plt.subplots(len(grids), 1, figsize=(16, len(grids) * 6), sharex=True)
    fig.suptitle(title, fontsize=20)

    for i, (operator, aggregated_values) in enumerate(aggregated_per_operator.items()):
        aggregated_values['color'] = aggregated_values[column].map(cmap_dict)
        aggregated_values.plot(ax=axes[i], color=aggregated_values['color'])
        axes[i].set_title(f"{operator}", fontsize=20)
        axes[i].set_ylim(bounds[1], bounds[3])
        axes[i].set_xlim(bounds[0], bounds[2])
        axes[i].set_axis_off()
        cx.add_basemap(axes[i], crs=aggregated_values.crs, source=map_source)

    # Create a legend for the colors
    patches = [mpatches.Patch(color=cmap_dict[val], label=str(val)) for val in sorted(unique_values)]
    fig.legend(handles=patches, title=TRANSLATIONS[column] if column in TRANSLATIONS else column, loc='center right', fontsize=20, title_fontsize=20)
    # Adjust the layout
    plt.tight_layout(rect=[0, 0, 0.83, 1])  # Leave space for the legend

    # Show the plot / save it
    if result_path:
        plt.savefig(result_path, dpi=500, bbox_inches='tight')
    if plot:
        plt.show()

    plt.close(fig)

def plot_continuous_values(grids, column, bounds, columns = None, map_source=cx.providers.OpenStreetMap.Mapnik, result_path=None, label="Number of Measurements", title="", plot=True, agg_func=count_rows_per_geometry):
    """Plot continuous values in a grid cell of a GeoDataFrame."""
    aggregated_per_operator = {}
    for operator, grid in grids.items():
        if columns:
            aggregated = agg_func(grid, columns)
        else:
            aggregated = agg_func(grid, column)
        aggregated_per_operator[operator] = aggregated

    # in case the colum is a list
    if isinstance(column, list):
        column = column[0]

    vmin = min(aggregated[column].min() for aggregated in aggregated_per_operator.values())
    vmax = max(aggregated[column].max() for aggregated in aggregated_per_operator.values())

    fig, axes = plt.subplots(len(grids), 1, figsize=(16, len(grids)*6), sharex=True)
    fig.suptitle(title, fontsize=20)
    for i, (operator, aggregated_per_operator) in enumerate(aggregated_per_operator.items()):
        # Plot the number of data points
        aggregated_per_operator.plot(column=column, cmap='viridis', legend=False, ax=axes[i], vmin=vmin, vmax=vmax)
        axes[i].set_title(f"{operator}", fontsize=20)
        axes[i].set_ylim(bounds[1], bounds[3])  # Set y-limits to the bounds
        axes[i].set_xlim(bounds[0], bounds[2]) # Set x-limits to the bounds
        axes[i].set_axis_off()
        cx.add_basemap(axes[i], crs=aggregated_per_operator.crs, source=map_source)

    # scale for all plots
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    sm = plt.cm.ScalarMappable(cmap='viridis', norm=norm)
    sm.set_array([])  # only needed for matplotlib < 3.6
    cbar = fig.colorbar(sm, ax=axes, orientation='horizontal', fraction=0.08, pad=0.04)
    cbar.ax.tick_params(labelsize=16)  # Set colorbar tick label size
    cbar.set_label(label, fontsize=20)

    # Show the plot
    if result_path:
        plt.savefig(result_path, dpi=500, bbox_inches='tight')
    if plot:
        plt.show()
    plt.close(fig)

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
    return network_availability

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

def calculate_bounds(gdfs):
    """Calculate the maximum bounds from a list of GeoDataFrames."""
    bounds = [gdf.total_bounds for gdf in gdfs]
    min_x = min(b[0] for b in bounds)
    min_y = min(b[1] for b in bounds)
    max_x = max(b[2] for b in bounds)
    max_y = max(b[3] for b in bounds)
    return min_x, min_y, max_x, max_y

def main():
    network_availability = load_network_availability_data(DATAROOT)
    network_latency = load_network_latency_data(DATAROOT)
    print("Evaluating network availability data...")
    rssi_min = np.inf
    rssi_max = -np.inf
    rsrp_min = np.inf
    rsrp_max = -np.inf
    rsrq_min = np.inf
    rsrq_max = -np.inf

    # Calculate bounds for the grid
    bounds = calculate_bounds(list(network_availability.values()) + list(network_latency.values()))

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
        grid = create_grid(gdf=gdf, n_cells=200, overlap=True, crs=gdf.crs, bounds=bounds)

        # Plot the number of data points in each grid cell
        plot_contnious_values(grid,
                              title="Number of Data Points in Grid Cells")
        # Plot the most common network availability value in each grid cell
        plot_categorical_values(grid, column='Typ',
                               color_mapping={'5G': 'green', '4G': 'yellow', '3G': 'orange', '2G': 'red',
                                               'kein Netz': 'black'}, title="Most Common Network Type in Grid Cells")
        # Plot the average network RSSI value in each grid cell
        plot_avg_value(grid, column='Signalstärke (RSSI) [dBm]',
                       title="Average RSSI in Grid Cells", scale_bounds=(rssi_min, rssi_max))
        # Plot the average network RSRP value in each grid cell
        plot_avg_value(grid, column='Signalstärke (RSRP) [dBm]',
                       title="Average RSRP in Grid Cells", scale_bounds=(rsrp_min, rsrp_max))
        # Plot the average network RSRQ value in each grid cell
        plot_avg_value(grid, column='Signalqualität (RSRQ) [dBm]',
                       title="Average RSRQ in Grid Cells", scale_bounds=(rsrq_min, rsrq_max))
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
        grid = create_grid(gdf=gdf, n_cells=200, overlap=True, crs=gdf.crs, bounds=bounds)

        plot_percentage_no_ping(grid, column=['ping_8.8.8.8', 'ping_1.1.1.1', 'ping_9.9.9.9'],
                                title=f"Percentage of Empty Pings in Grid Cells ({operator})")
        plot_no_pings(grid, column=['ping_8.8.8.8', 'ping_1.1.1.1', 'ping_9.9.9.9'],
                      title=f"Number of Empty Pings in Grid Cells ({operator})")
        # Plot the number of data points in each grid cell

        plot_contnious_values(grid, column='timestamp',
                              title="Number of Data Points in Grid Cells")
        # Plot the average network latency in each grid cell
        plot_avg_value(grid, column='ping_8.8.8.8',
                       title="Average Latency to ping_8.8.8.8 in Grid Cells in ms")

        plot_avg_value(grid, column=['ping_1.1.1.1', 'ping_8.8.8.8', 'ping_9.9.9.9'],
                       title="Average Latency in Grid Cells in ms", scale_bounds=(latency_min, latency_max))

def test():
    network_availability = load_network_availability_data(DATAROOT)
    network_latency = load_network_latency_data(DATAROOT)
    bounds = calculate_bounds(list(network_availability.values()) + list(network_latency.values()))

    grids_availability = {}
    grids_latency = {}
    for operator, gdf in network_availability.items():
        grid = create_grid(gdf=gdf, n_cells=200, overlap=True, crs=gdf.crs, bounds=bounds)
        grids_availability[operator] = grid
    for operator, gdf in network_latency.items():
        grid = create_grid(gdf=gdf, n_cells=200, overlap=True, crs=gdf.crs, bounds=bounds)
        grids_latency[operator] = grid

    # Plot the number of data points in each grid cell
    plot_continuous_values(grids_availability, "Messpunkt", bounds, title="Number of Data Points in Grid Cells (Availability)", agg_func=count_rows_per_geometry)
    plot_continuous_values(grids_latency, "timestamp", bounds, title="Number of Data Points in Grid Cells (Latency)", agg_func=count_rows_per_geometry)
    # Plot the most common network availability value in each grid cell
    #plot_categorical_values(grids_availability, bounds, column='Typ', title="Most Common Network Type in Grid Cells", agg_func=calculate_most_common_network_per_geometry)
    # plot the worst network availability value in each grid cell
    #plot_categorical_values(grids_availability, bounds, column='Typ', title="Most Common Network Type in Grid Cells", agg_func=calculate_worst_network_per_geometry)
    # plot the average RSSI value in each grid cell
    #plot_continuous_values(grids_availability, "Signalstärke (RSSI) [dBm]", bounds, title="Average RSSI in Grid Cells", agg_func=calculate_avg_value_per_geometry, label="RSSI [dBm]")
    # plot the average RSRP value in each grid cell
    #plot_continuous_values(grids_availability, "Signalstärke (RSRP) [dBm]", bounds, title="Average RSRP in Grid Cells", agg_func=calculate_avg_value_per_geometry, label="RSRP [dBm]")
    # plot the average RSRQ value in each grid cell
    #plot_continuous_values(grids_availability, "Signalqualität (RSRQ) [dBm]", bounds, title="Average RSRQ in Grid Cells", agg_func=calculate_avg_value_per_geometry, label="RSRQ [dBm]")

    # plot the average latency to all pings in each grid cell
    plot_continuous_values(grids_latency, 'mean', bounds, columns=PINGS ,title="Average Latency in Grid Cells", agg_func=calculate_avg_value_per_geometry_multiple_columns, label="Latency [ms]")

if __name__ == "__main__":
    #main()
    test()

...