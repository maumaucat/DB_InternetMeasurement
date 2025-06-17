import math
from pathlib import Path

import matplotlib.pyplot as plt
import shapely
import matplotlib.patches as mpatches
import contextily as cx
import matplotlib.colors as mcolors
from matplotlib.colors import BoundaryNorm

from Aggregation import *

PLOT = True
DATAROOT = Path('data')
OPERATORS = ['Vodafone', 'Telekom', 'o2']
TRANSLATIONS = { 'Typ' : 'Type'
                 ,'Netzwerk Anbieter' : 'Network Provider'}

PINGS = ['ping_8.8.8.8', 'ping_1.1.1.1', 'ping_9.9.9.9', 'ping_google.com', 'ping_utwente.nl']


def create_grid(gdf=None,
                bounds=None,
                cell_size_degree=0.01,
                overlap=False,
                crs="EPSG:29902"):
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
    cell_size = cell_size_degree #(xmax-xmin)/n_cells
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

def plot_categorical_values(grids,
                            bounds,
                            column,
                            map_source=cx.providers.OpenStreetMap.Mapnik,
                            result_path=None,
                            title="",
                            plot=PLOT,
                            agg_func=calculate_most_common_network_per_geometry,
                            one_legend=True):
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
                           plot=PLOT,
                           agg_func=count_rows_per_geometry,
                           log_scale=False,
                           scale_space=None,):
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
    if scale_space is not None:
        ticks = scale_space
        cmap = plt.get_cmap('viridis', len(scale_space) -1)
        norm = BoundaryNorm(boundaries=scale_space, ncolors=cmap.N)
    else:
        cmap = plt.get_cmap('viridis')
        if log_scale:
            norm = mcolors.LogNorm(vmin=vmin, vmax=vmax)
            ticks = np.logspace(np.log10(vmin), np.log10(vmax), num=15)
        else:
            norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
            ticks = np.linspace(vmin, vmax, num=15)

    ticks = np.round(ticks, 0)

    fig, axes = plt.subplots(len(grids), 1, figsize=(16, len(grids)*6), sharex=True)
    fig.suptitle(title, fontsize=20)
    for i, (operator, aggregated) in enumerate(aggregated_per_operator.items()):

        # Plot the number of data points
        aggregated.plot(column=column, cmap=cmap, legend=False, ax=axes[i], norm=norm)
        axes[i].set_title(f"{operator}", fontsize=20)
        axes[i].set_ylim(bounds[1], bounds[3])  # Set y-limits to the bounds
        axes[i].set_xlim(bounds[0], bounds[2]) # Set x-limits to the bounds
        axes[i].set_axis_off()
        cx.add_basemap(axes[i], crs=aggregated.crs, source=map_source)


    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
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
                           plot=PLOT,
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
        axes[i][0].set_xticks([])  # remove x-numbers
        axes[i][0].tick_params(axis='x', length=0)  # no ticks on x-axis
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


def plot_bars(grids,
              column,
              result_path=None,
              ylabel="",
              xlabel="",
              title="",
              plot=PLOT,
              agg_func=None):
    """Plot bar charts for categorical values in a grid cell of a GeoDataFrame."""
    aggregated_per_operator = {}
    for operator, grid in grids.items():
        aggregated = agg_func(grid, column)
        aggregated_per_operator[operator] = aggregated

    all_categories = set()
    for aggregated in aggregated_per_operator.values():
        all_categories.update(aggregated[column].unique())
    all_categories = sorted(all_categories)

    plot_df = pd.DataFrame(index=all_categories)

    for operator, agg in aggregated_per_operator.items():
        # Prozentwerte auf die Verbindungstypen abbilden, fehlende Werte mit 0 füllen
        operator_series = agg.set_index('Typ')['percentage'].reindex(all_categories, fill_value=0)
        plot_df[operator] = operator_series

    # Plot
    ax = plot_df.plot(kind='bar', figsize=(12, 7))
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    plt.xticks(rotation=0)

    if result_path:
        plt.savefig(result_path, dpi=500)
    if plot:
        plt.show()
    plt.close()

def plot_avg_latency_per_ping_bar(grids,
                                  ping_columns,
                                  ylabel="Average Latency [ms]",
                                  xlabel="Provider",
                                  title="",
                                  result_path=None,
                                  plot=PLOT):
    """Plot average latency per ping in a bar chart for each operator."""

    plot_df = pd.DataFrame(index=grids.keys(), columns=ping_columns)

    for operator, grid in grids.items():
        for ping_col in ping_columns:
            if ping_col in grid.columns:
                avg_latency = grid[ping_col].mean()
                plot_df.at[operator, ping_col] = avg_latency
            else:
                plot_df.at[operator, ping_col] = float('nan')

    plot_df = plot_df.astype(float)

    # transponieren
    plot_df = plot_df.T

    ax = plot_df.plot(kind='bar', figsize=(12, 7))
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    plt.xticks(rotation=0)
    plt.legend(title="Provider")

    if result_path:
        plt.savefig(result_path, dpi=500)
    if plot:
        plt.show()
    plt.close()

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
    min_x = min(b[0] for b in bounds) - 0.01  # add a small margin
    min_y = min(b[1] for b in bounds) - 0.01  # add a small margin
    max_x = max(b[2] for b in bounds) + 0.01  # add a small margin
    max_y = max(b[3] for b in bounds) + 0.01  # add a small margin
    return min_x, min_y, max_x, max_y

def main():
    network_availability = load_network_availability_data()
    network_latency = load_network_latency_data()
    bounds = calculate_bounds(list(network_availability.values()) + list(network_latency.values()))

    grids_availability = {}
    grids_latency = {}
    for operator, gdf in network_availability.items():
        grid = create_grid(gdf=gdf, cell_size_degree=0.01, overlap=True, crs=gdf.crs, bounds=bounds)
        grids_availability[operator] = grid
    for operator, gdf in network_latency.items():
        grid = create_grid(gdf=gdf, cell_size_degree=0.01, overlap=True, crs=gdf.crs, bounds=bounds)
        grids_latency[operator] = grid


    # Ensure plt.savefig does not throw errors for not existing parent folders by creating them
    for dir in ["numdatapoints", "networktype", "signal_strength", "signal_quality"]:
        Path(f"plots/{dir}").mkdir(parents=True)

    """ NUMBER OF DATA POINTS """
    # Plot the number of data points in each grid cell
    plot_continuous_values(grids_availability,
                           "Messpunkt",
                           bounds,
                           title="Number of Data Points in Grid Cells (Availability)",
                           agg_func=count_rows_per_geometry,
                           result_path="plots/numdatapoints/availability_data_points.svg",
                           scale_space=[0, 5, 10, 20, 30, 50, 100, 150, 200, 250])
    plot_continuous_values(grids_latency,
                           "timestamp",
                           bounds,
                           title="Number of Data Points in Grid Cells (Latency)",
                           agg_func=count_rows_per_geometry,
                           result_path="plots/numdatapoints/latency_data_points.svg",
                           scale_space=[0, 5, 10, 20, 30, 50, 100, 150, 200, 250])
    """ NETWORK TYPES """
    # Plot the most common network availability value in each grid cell
    plot_categorical_values(grids_availability,
                            bounds,
                            column='Typ',
                            title="Most Common Network Type in Grid Cells",
                            agg_func=calculate_most_common_network_per_geometry,
                            result_path="plots/networktype/most_common_network_type.svg")
    # plot the worst network availability value in each grid cell
    plot_categorical_values(grids_availability,
                            bounds,
                            column='Typ',
                            title="Worst Network Type in Grid Cells",
                            agg_func=calculate_worst_network_per_geometry,
                            result_path="plots/networktype/worst_network_type.svg")

    # plot the percentage of each network type overall
    plot_bars(grids_availability,
              'Typ',
              title="Percentage of Network Types overall",
              agg_func=calculate_percentage_overall,
              xlabel="Provider",
              ylabel="Percentage",
              result_path="plots/networktype/percentage_network_types.svg")

    """ SIGNAL STRENGTH """
    # plot the average RSSI value in each grid cell
    plot_continuous_values(grids_availability,
                           "Signalstärke (RSSI) [dBm]",
                           bounds,
                           title="Average RSSI in Grid Cells",
                           agg_func=calculate_avg_value_per_geometry,
                           label="RSSI [dBm]",
                           result_path="plots/signal_strength/average_rssi.svg")
    # plot the average RSRP value in each grid cell
    plot_continuous_values(grids_availability,
                           "Signalstärke (RSRP) [dBm]",
                           bounds,
                           title="Average RSRP in Grid Cells",
                           agg_func=calculate_avg_value_per_geometry,
                           label="RSRP [dBm]",
                           result_path="plots/signal_strength/average_rsrp.svg")
    # plot the average RSRQ value in each grid cell
    plot_continuous_values(grids_availability,
                           "Signalqualität (RSRQ) [dBm]",
                           bounds,
                           title="Average RSRQ in Grid Cells",
                           agg_func=calculate_avg_value_per_geometry,
                           label="RSRQ [dBm]",
                           result_path="plots/signal_quality/average_rsrq.svg")
    # plot the average RSSI value per connection type in each grid cell
    plot_continuous_values_per_Type(grids_availability,
                                    'Signalstärke (RSSI) [dBm]',
                                    bounds,
                                    title="Average RSSI per Connection Type in Grid Cells",
                                    agg_func=calculate_avg_value_per_networktype,
                                    label="RSSI [dBm]",
                                    types=['4G', '5G'],
                                    result_path="plots/signal_strength/average_rssi_per_type.svg")
    # plot the average RSRP value per connection type in each grid cell
    plot_continuous_values_per_Type(grids_availability,
                                    'Signalstärke (RSRP) [dBm]',
                                    bounds,
                                    title="Average RSRP per Connection Type in Grid Cells",
                                    agg_func=calculate_avg_value_per_networktype,
                                    label="RSRP [dBm]",
                                    types=['4G', '5G'],
                                    result_path="plots/signal_strength/average_rsrp_per_type.svg")

    # plot the average RSRQ value per connection type in each grid cell
    plot_continuous_values_per_Type(grids_availability,
                                    'Signalqualität (RSRQ) [dBm]',
                                    bounds,
                                    title="Average RSRQ per Connection Type in Grid Cells",
                                    agg_func=calculate_avg_value_per_networktype,
                                    label="RSRQ [dBm]",
                                    types=['4G', '5G'],
                                    result_path="plots/signal_quality/average_rsrq_per_type.svg")

    # plot the median RSSI value in each grid cell
    plot_continuous_values(grids_availability,
                            "Signalstärke (RSSI) [dBm]",
                            bounds,
                            title="Median RSSI in Grid Cells",
                            agg_func=calculate_median_value_per_geometry,
                            label="RSSI [dBm]",
                            result_path="plots/signal_strength/median_rssi.svg")
    # plot the median RSRP value in each grid cell
    plot_continuous_values(grids_availability,
                            "Signalstärke (RSRP) [dBm]",
                            bounds,
                            title="Median RSRP in Grid Cells",
                            agg_func=calculate_median_value_per_geometry,
                            label="RSRP [dBm]",
                            result_path="plots/signal_strength/median_rsrp.svg")
    # plot the median RSRQ value in each grid cell
    plot_continuous_values(grids_availability,
                            "Signalqualität (RSRQ) [dBm]",
                            bounds,
                            title="Median RSRQ in Grid Cells",
                            agg_func=calculate_median_value_per_geometry,
                            label="RSRQ [dBm]",
                            result_path="plots/signal_quality/median_rsrq.svg")


    """LATENCY"""
    # plot the average latency to all pings in each grid cell
    plot_continuous_values(grids_latency,
                           'mean',
                           bounds,
                           columns=PINGS,
                           title="Average Latency in Grid Cells",
                           agg_func=calculate_avg_value_per_geometry_multiple_columns,
                           label="Latency [ms]",
                           log_scale=True,
                           result_path="plots/latency/average_latency.svg")
    # plot the median latency to all pings in each grid cell
    plot_continuous_values(grids_latency,
                           'median',
                           bounds,
                           columns=PINGS,
                           title="Median Latency in Grid Cells",
                           agg_func=calculate_median_value_per_geometry_multiple_columns,
                           label="Latency [ms]",
                           log_scale=True,
                           result_path="plots/latency/median_latency.svg")
    # plot the percentage of empty pings in each grid cell
    plot_continuous_values(grids_latency,
                           'percentage_empty',
                           bounds,
                           columns=PINGS,
                           title="Percentage of Empty Pings in Grid Cells",
                           agg_func=percentage_of_empty_per_geometry,
                           label="Percentage of Empty Pings",
                           result_path="plots/latency/percentage_empty_pings.svg")
    # plot the average latency per ping in a bar chart
    plot_avg_latency_per_ping_bar(grids_latency,
                                  PINGS,
                                  title="Average Latency per Ping",
                                  xlabel="Provider",
                                  ylabel="Average Latency [ms]",
                                  result_path="plots/latency/average_latency_per_ping.svg")

    """ NETWORK PROVIDER """
    # plot the provider
    plot_categorical_values(grids_availability,
                            bounds,
                            column='Netzwerk Anbieter',
                            title="Network Provider",
                            agg_func=calculate_all_per_geometry,
                            one_legend=False,
                            result_path="plots/network_provider/network_provider.svg")

if __name__ == "__main__":
    main()

...
