from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shapely
import matplotlib.patches as mpatches
import contextily as cx
import matplotlib.colors as mcolors
from odc.geo.geobox import rotate

from Aggregation import *

DATAROOT = Path('data')
OPERATORS = ['Vodafone', 'Telekom', 'o2']
TRANSLATIONS = { 'Typ' : 'Type'
                 ,'Netzwerk Anbieter' : 'Network Provider'}

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

def plot_categorical_values(grids, bounds, column, map_source=cx.providers.OpenStreetMap.Mapnik, result_path=None, title="", plot=True, agg_func=calculate_most_common_network_per_geometry, one_legend=True):
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
        if not one_legend:
            # Echte, vorkommende Kategorien im Plot (ohne NaN)
            present_vals = sorted(aggregated_values[column].dropna().unique())
            # Patches nur für vorhandene Kategorien
            patches = [mpatches.Patch(color=cmap_dict[val], label=str(val)) for val in present_vals]

            axes[i].legend(handles=patches, title=TRANSLATIONS.get(column, column), fontsize=12, loc='upper right')
        axes[i].set_title(f"{operator}", fontsize=20)
        axes[i].set_ylim(bounds[1], bounds[3])
        axes[i].set_xlim(bounds[0], bounds[2])
        axes[i].set_axis_off()
        cx.add_basemap(axes[i], crs=aggregated_values.crs, source=map_source)

    # Create a legend for the colors
    if one_legend:
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

def plot_continuous_values(grids,
                           column,
                           bounds,
                           columns = None,
                           map_source=cx.providers.OpenStreetMap.Mapnik,
                           result_path=None,
                           label="Number of Measurements",
                           title="",
                           plot=True,
                           agg_func=count_rows_per_geometry,
                           log_scale=False):
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
    # scale for all plots
    if log_scale:
        norm = mcolors.LogNorm(vmin=vmin, vmax=vmax)
        ticks = np.logspace(np.log10(vmin), np.log10(vmax), num=15)
        # round ticks to 0 decimal places
        ticks = np.round(ticks, 0)
    else:
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        ticks = np.linspace(vmin, vmax, num=15)
        ticks = np.round(ticks, 0)

    fig, axes = plt.subplots(len(grids), 1, figsize=(16, len(grids)*6), sharex=True)
    fig.suptitle(title, fontsize=20)
    for i, (operator, aggregated) in enumerate(aggregated_per_operator.items()):
        # Plot the number of data points
        aggregated.plot(column=column, cmap='viridis', legend=False, ax=axes[i], norm=norm)
        axes[i].set_title(f"{operator}", fontsize=20)
        axes[i].set_ylim(bounds[1], bounds[3])  # Set y-limits to the bounds
        axes[i].set_xlim(bounds[0], bounds[2]) # Set x-limits to the bounds
        axes[i].set_axis_off()
        cx.add_basemap(axes[i], crs=aggregated.crs, source=map_source)


    sm = plt.cm.ScalarMappable(cmap='viridis', norm=norm)
    sm.set_array([])  # only needed for matplotlib < 3.6

    cbar = fig.colorbar(sm, ax=axes, orientation='horizontal', fraction=0.08, pad=0.04, ticks=ticks)
    cbar.set_ticklabels([f"{tick:.0f}" for tick in ticks])  # Format tick labels
    cbar.ax.tick_params(labelsize=16)  # Set colorbar tick label size
    cbar.set_label(label, fontsize=20)

    # Show the plot
    if result_path:
        plt.savefig(result_path, dpi=500, bbox_inches='tight')
    if plot:
        plt.show()
    plt.close(fig)


def plot_continuous_values_per_Type(grids,
                           column,
                           bounds,
                           map_source=cx.providers.OpenStreetMap.Mapnik,
                           result_path=None,
                           label="Number of Measurements",
                           title="",
                           plot=True,
                           agg_func=count_rows_per_geometry,
                           log_scale=False,
                           types=NETWORKTYPES):
    """Plot continuous values in a grid cell of a GeoDataFrame, grouped by Type."""
    aggregated_per_operator = {}
    for operator, grid in grids.items():
        aggregated = agg_func(grid, column)
        aggregated_per_operator[operator] = aggregated

    vmin = min(aggregated[column].min() for aggregated in aggregated_per_operator.values())
    vmax = max(aggregated[column].max() for aggregated in aggregated_per_operator.values())

    # scale for all plots
    if log_scale:
        norm = mcolors.LogNorm(vmin=vmin, vmax=vmax)
        ticks = np.logspace(np.log10(vmin), np.log10(vmax), num=15)
        # round ticks to 0 decimal places
        ticks = np.round(ticks, 0)
    else:
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        ticks = np.linspace(vmin, vmax, num=15)
        ticks = np.round(ticks, 0)

    fig, axes = plt.subplots(len(grids), len(types), figsize=(16*len(types), len(grids)*6), sharex=True)
    fig.suptitle(title, fontsize=20)
    for i in range(len(types)):
        axes[0][i].set_title(types[i], fontsize=20)
    for i, (operator, aggregated) in enumerate(aggregated_per_operator.items()):
        for j, network_type in enumerate(types):
            # Filter the aggregated data for the current network type
            filtered_aggregated = aggregated[aggregated['Typ'] == network_type]
            if filtered_aggregated.empty:
                raise ValueError(f"No data found for operator {operator} and network type {network_type}. Please check your data.")

            filtered_aggregated.plot(column=column, cmap='viridis', legend=False, ax=axes[i][j], norm=norm)
            axes[i][j].set_ylim(bounds[1], bounds[3])
            axes[i][j].set_xlim(bounds[0], bounds[2])
            axes[i][j].set_axis_off()
            cx.add_basemap(axes[i][j], crs=filtered_aggregated.crs, source=map_source)

        axes[i][0].set_ylabel(f"{operator}", fontsize=20)
        axes[i][0].set_axis_on()
        axes[i][0].set_yticks([])  # remove y-numbers
        axes[i][0].tick_params(axis='y', length=0)  # no ticks on y-axis

    sm = plt.cm.ScalarMappable(cmap='viridis', norm=norm)
    sm.set_array([])  # only needed for matplotlib < 3.6
    cbar = fig.colorbar(sm, ax=axes, orientation='horizontal', fraction=0.08, pad=0.04, ticks=ticks)
    cbar.set_ticklabels([f"{tick:.0f}" for tick in ticks])  # Format tick labels
    cbar.ax.tick_params(labelsize=16)  # Set colorbar tick label size
    cbar.set_label(label, fontsize=20)

    # Show the plot
    if result_path:
        plt.savefig(result_path, dpi=500)
    if plot:
        plt.show()



def load_network_availability_data():
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

def load_network_latency_data():
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
    network_availability = load_network_availability_data()
    network_latency = load_network_latency_data()
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
    #plot_continuous_values(grids_availability, "Messpunkt", bounds, title="Number of Data Points in Grid Cells (Availability)", agg_func=count_rows_per_geometry)
    #plot_continuous_values(grids_latency, "timestamp", bounds, title="Number of Data Points in Grid Cells (Latency)", agg_func=count_rows_per_geometry)
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
    #plot_continuous_values(grids_availability, "Signalqualität (RSRQ) [dBm]", bounds, title="Median SINR in Grid Cells", agg_func=calculate_median_value_per_geometry, label="RSRQ [dBm]")
    # plot the average latency to all pings in each grid cell
    #plot_continuous_values(grids_latency, 'mean', bounds, columns=PINGS ,title="Average Latency in Grid Cells", agg_func=calculate_avg_value_per_geometry_multiple_columns, label="Latency [ms]", log_scale=True)
    # plot the median latency to all pings in each grid cell
    #plot_continuous_values(grids_latency, 'median', bounds, columns=PINGS, title="Median Latency in Grid Cells", agg_func=calculate_median_value_per_geometry_multiple_columns, label="Latency [ms]", log_scale=True)
    # plot the percentage of empty pings in each grid cell
    #plot_continuous_values(grids_latency, 'percentage_empty', bounds, columns=PINGS, title="Percentage of Empty Pings in Grid Cells", agg_func=percentage_of_empty_per_geometry, label="Percentage of Empty Pings", log_scale=False)
    # plot the provider
    # plot_categorical_values(grids_availability, bounds, column='Netzwerk Anbieter', title="Network Provider", agg_func=calculate_all_per_geometry, one_legend=False)
    # plot the latency per connection type
    plot_continuous_values_per_Type(grids_availability, 'Signalstärke (RSSI) [dBm]', bounds, title="Average RSSI per Connection Type in Grid Cells", agg_func=calculate_avg_value_per_networktype, label="RSSI [dBm]", log_scale=False, types=['4G', '5G'])

if __name__ == "__main__":
    main()

...