def keep_regs(df, regs):
    """ Example function. Keep only the subset regs of regions in data.

    Args:
        df (pd.DataFrame): pandas dataframe 

    Returns:
        df (pd.DataFrame): pandas dataframe

    """ 
    
    for r in regs:
        I = df.reg.str.contains(r)
        df = df.loc[I == False] # keep everything else
    
    return df



#_________________________PLOT 1______________________________________ 

import pandas as pd
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display

def display_air_traffic(airt):
    # List of non-country regions to exclude generally
    non_countries = [
        'World', 'EU', 'High income', 'OECD members', 'North America', 'Europe & Central Asia',
        'IDA & IBRD total', 'Low & middle income', 'IBRD only', 'Middle income', 'Upper middle income',
        'Late-demographic dividend', 'East Asia & Pacific', 'East Asia & Pacific (IDA & IBRD countries)',
        'East Asia & Pacific (excluding high income)', 'Early-demographic dividend', 'European Union',
        'Euro area', 'Arab World', 'South Asia', 'IDA total'
    ]

    # List of regions to specifically include despite general exclusion criteria
    include_specific = ['United Arab Emirates', 'Hong Kong SAR, China', "Korea, Dem. People's Rep."]

    # Adjust the exclusion list by removing the entries we want to include
    adjusted_non_countries = [region for region in non_countries if region not in include_specific]

    # Transposing the dataframe to have years as rows and setting the index
    plot_data = airt.set_index('Country Name').T

    # Identify entries to exclude based on being a non-country or having a name longer than 18 characters
    excluded_entries = adjusted_non_countries + [name for name in plot_data.columns if len(name) > 18]

    # Print excluded entries
    print("Excluded entries:", excluded_entries)

    # Filter out non-country regions and long name countries from the DataFrame
    plot_data = plot_data.drop(excluded_entries, axis=1, errors='ignore')

    # Function to plot the top 10 countries for a selected year
    def plot_top_countries(year):
        if year in plot_data.index:  # Check if the year is valid after filtering
            top_countries = plot_data.loc[year].nlargest(10)  # Get the 10 countries with the most passengers
            plt.figure(figsize=(10, 6))
            top_countries.plot(kind='bar', color='skyblue')
            plt.title(f'Top 10 Countries by Air Traffic Passengers in {year}')
            plt.xlabel('Country')
            plt.ylabel('Number of Passengers')
            plt.xticks(rotation=45)
            plt.grid(True, which='both', linestyle='--', linewidth=0.5)
            plt.show()
        else:
            print("Selected year is not available in the dataset.")

    # Dropdown widget to select the year
    year_dropdown = widgets.Dropdown(
        options=plot_data.index,
        value=plot_data.index[-1] if not plot_data.index.empty else None,
        description='Select Year:',
        disabled=False,
    )

    # Display the widget and the updating plot
    widgets.interact(plot_top_countries, year=year_dropdown)

#_______________________________________________________________ 


#_________________________PLOT 2______________________________________ 

def display_air_traffic_interactive(airt):
    import matplotlib.pyplot as plt
    import ipywidgets as widgets
    from IPython.display import display
    
    # Transposing the dataframe to have years as rows and setting the index
    plot_data = airt.set_index('Country Name').T

    # Correcting the index by removing non-numeric characters and converting to integer
    plot_data.index = plot_data.index.str.extract('(\d+)', expand=False).astype(int)

    # Ensure the dropdown's default value is within range
    default_country_index = min(259, len(plot_data.columns) - 1)
    country_dropdown = widgets.Dropdown(
        options=plot_data.columns,
        value=plot_data.columns[default_country_index],
        description='Country:',
        disabled=False,
    )

    # Plotting function that takes a country name as input
    def plot_air_traffic(country):
        plt.figure(figsize=(10, 6))
        plot_data[country].dropna().plot(marker='o')
        plt.title(f'Air Traffic Passengers for {country} (1970-2021)')
        plt.xlabel('Year')
        plt.ylabel('Number of Passengers')
        plt.grid(True)
        plt.show()

    # Interactive widget to select a country and display the graph
    widgets.interact(plot_air_traffic, country=country_dropdown)
#_______________________________________________________________ 


#_________________________PLOT 3______________________________________ 

def display_air_traffic_summarize(airt):
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np
    from statsmodels.tsa.arima.model import ARIMA
    import ipywidgets as widgets
    from IPython.display import display

    # Assuming 'airt' is your DataFrame and is structured correctly
    plot_data = airt.set_index('Country Name').T

    # Correcting the index by stripping the 'y' prefix and converting to integers
    plot_data.index = plot_data.index.str.strip('y').astype(int)

    #   Extracting data for 'World'
    world_data = plot_data['World']

    # Fit an ARIMA model (example parameters used, adjust according to your data)
    model = ARIMA(world_data.dropna(), order=(1, 1, 1))
    fitted_model = model.fit()

    #    Forecast the next 5 periods
    forecast = fitted_model.forecast(steps=5)

    # Creating a figure and axis object
    fig, ax1 = plt.subplots(figsize=(14, 7))

    # Plot historical data
    world_data.plot(ax=ax1, color='blue', marker='o', label='Historical Data')

    # Plot forecasted data
    last_year = world_data.index[-1]  # Now this is already an integer
    forecast_years = np.arange(last_year + 1, last_year + 6)  # Now this should work without error
    ax1.plot(forecast_years, forecast, color='red', marker='x', linestyle='dashed', label='Forecast')

    # Creating a second y-axis for growth rate
    ax2 = ax1.twinx()
    annual_growth_rate = world_data.pct_change() * 100
    ax2.plot(annual_growth_rate.index, annual_growth_rate, color='green', marker='*', linestyle='-', label='Annual Growth Rate')

    # Formatting the plot
    ax1.set_title('Total Air Traffic Passengers (World) with Forecast')
    ax1.set_xlabel('Year')
    ax1.set_ylabel('Number of Passengers', color='blue')
    ax2.set_ylabel('Growth Rate (%)', color='green')
    ax1.grid(True)

    # Handling legends
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='upper left')

    plt.show()

#__________________________________________________________________________

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd

def plot_heatmap():
    # Load the geographic data
    world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))

    # Load your data from the Excel file
    data = pd.read_excel('airtraffic.xlsx')

    # Merge the geographic data with your data
    world = world.merge(data, how="left", left_on="iso_a3", right_on="Country Code")

     # Manually setting the minimum and maximum values for the color scale
    vmin = 0  # Minimum value of the color scale
    vmax = 1000000000  # Maximum value of the color scale

    # Plotting
    fig, ax = plt.subplots(1, 1, figsize=(15, 10))
    world.boundary.plot(ax=ax, linewidth=1)  # Draw boundary lines
    world.plot(column='2019', ax=ax, legend=True,
               legend_kwds={'label': "Billion passengers in 2019", 'orientation': "horizontal"},
               cmap='Reds', vmin=vmin, vmax=vmax)  # Using 'Reds' colormap

    # Hide all axis details
    ax.set_xticks([])
    ax.set_yticks([])
    ax.xaxis.set_visible(False)  # Ensure x-axis is not visible
    ax.yaxis.set_visible(False)  # Ensure y-axis is not visible

    # Add a title to the plot
    ax.set_title("Flight Passengers in 2019", fontsize=16, fontweight='bold')

    plt.show()


