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
