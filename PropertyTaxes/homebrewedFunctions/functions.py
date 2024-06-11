import time
import numpy as np
import pandas as pd
import pandas_datareader.data as web
import pingouin
import matplotlib.pyplot as plt
import datetime
from linearmodels import PanelOLS
from linearmodels.panel import compare

class info_criterion():
    ## Thank you Abi Idowu
    # Function to calculate AIC
    def calculate_aic(self, n, rss, k):
        return n * np.log(rss / n) + 2 * k
    
    # Function to calculate BIC
    def calculate_bic(self, n, rss, k):
        return n * np.log(rss / n) + k * np.log(n)
    
    # Function to calculate HQIC
    def calculate_hqic(self, n, rss, k):
        return n * np.log(rss / n) + 2 * k * np.log(np.log(n))

def gather_data(data_codes, 
                start, 
                end = datetime.datetime.today(), 
                freq = "M"):
    i = 0
    # dct.items() calls key and value that key points to
    for key, val in data_codes.items():
        time.sleep(.51)
        if i == 0:
            # Create dataframe for first variable, then rename column
            df = web.DataReader(
                val, "fred", start, end).resample(freq).mean()
            df.rename(columns = {val:key}, inplace = True) 
            # setting i to None will cause the next block of code to execute,
            # placing data within df instead of creating a new dataframe for
            # each variable
            i = None
        else:
            # If dataframe already exists, add new column
            df[key] = web.DataReader(val, "fred", start, end).resample(freq).mean()

    return df

def save_dict_of_dfs_to_excel(dict_of_dfs, filename):
    """
    Save a dictionary of DataFrames to an Excel file, with each DataFrame in a separate sheet.

    Parameters:
    dict_of_dfs: dict # A dictionary where keys are sheet names and values are DataFrames.
    filename: str # The path to the output Excel file.
    """
    with pd.ExcelWriter(filename) as writer:
        for sheet_name, df in dict_of_dfs.items():
            df.reset_index().to_excel(writer, sheet_name=sheet_name)
    print(f"File saved as {filename}")

def plot_scatter_corr(plot_df, title, corr="corr", **kwargs):

    """
    Parameters:
    plot_df: pd.DataFrame
    title: str
    corr: str
    **kwargs: dict # matplotlib kwargs
    """
    
    corr_df = getattr(plot_df, corr)().round(2)
    num_keys = len(plot_df.keys())
    fig, axs = plt.subplots(num_keys, num_keys, figsize=(20,20))
    
    for i, key1 in enumerate(plot_df.keys()):
        for j, key2 in enumerate(plot_df.keys()):
            
            if i < j:
                ax = axs[j][i]
                plot_df.plot.scatter(x=key1, y=key2, ax=ax, **kwargs)
                # Remove color bar label if 'c' is in kwargs
                if 'c' in kwargs:
                    cbar = ax.collections[0].colorbar
                    if cbar is not None:
                        cbar.set_label('')
                        cbar.set_ticklabels([])
            elif i > j:
                ax = axs[j][i]
                ax.text(.5, .5, corr_df.astype(str).loc[key2, key1],
                        ha="center", va="center", fontsize=28)
                ax.set_xticklabels([])
            else:
                ax = axs[i][j]
                plot_df[[key1]].hist(density=True, ax=ax)
                if i == 0: ax.set_ylabel(key1)
            
            if i > 0:
                ax.set_ylabel("")
                ax.set_yticklabels([])
            else:
                ax.set_ylabel(key2.replace(" ", "\n"), fontsize=24)
            if j < num_keys - 1: 
                ax.set_xticklabels([])
            if j > 0:
                ax.set_title("")
            else:
                ax.set_title(key1.replace(" ", "\n"), fontsize=24)
            ax.set_xlabel("")
    
    plt.suptitle(title, fontsize=40, y=1.025)
    return fig, ax



def compare_regs_plot(compare_regs,y, variant="", title =""):

    """
    Parameters:
    compare_regs : {} # dictionary of regression results (statsmodels or linear models)
    y : str # string indicated the dependent variable from the set of regressions to be compared
    variant : str # string that used to indicate which subset of regresssions is reflected in compare_regs
    title : str # string to be used as base for title of plots that indicate results
    """
    
    title = f"y = {y}" if title == "" else title
    hlines = {"params":[0],
              "tstats":[0],
              "pvalues":[0.05, 0.1]}
    fig, axs = plt.subplots(2, 1, figsize = (20,6))
    stat_names = ("params", "pvalues")
    for i, stat in enumerate(stat_names):
        ax = axs[i]
        getattr(compare_regs, stat).plot.bar(ax = axs[i], legend = False)
        ax.set_ylabel(stat.title())
        
        if i == 0:
            ax.legend(loc = 1, ncols = 4, bbox_to_anchor = (.85,1.43), fontsize = 18)
        if i < len(stat_names) - 1:
            ax.set_xticklabels([])
            ax.set_xlabel("")
        else:
            ax.set_xticklabels([x.replace(" ", "\n") for x in compare_regs.params.index], rotation = "horizontal", fontsize = 20)

        for yval in hlines[stat]:
            ax.axhline(y = yval, ls = "--", 
                       linewidth = 2, color = "k")

        plt.suptitle(f"{variant}: {title}", y = 1.2)
    return fig, axs
def plot_r2(r2_df, r2s, key, variant):

    """
    r2_df : # dataframe generated by gathering results of regressions to be compared. Each column refers to a different regression to be compared.
    r2s : list # list of strings for uses in this project, indicates r2, within r2, and between r2
    key : str # refers to y variable compared across regressions
    variant : str # string that used to indicate which subset of regresssions is reflected in compare_regs
    """
    
    num_plots = 3
    fig, axs = plt.subplots(num_plots, 1, figsize = (20,10))
    for n, r2 in enumerate(r2s):
        ax = axs[n]
        plot_df = r2_df[r2_df["r2"] == r2]
        plot_df.plot(x = "y", y = list(r2_df.keys())[2:], kind = "bar", legend = False, ax = ax)
        if n == 0: 
            ax.legend(loc = 1, ncols = 4, bbox_to_anchor = (.85,1.4), fontsize = 18)        
        if n + 1 == num_plots: 
            ax.set_xticklabels([x for x in ax.get_xticklabels()], 
                               rotation = "horizontal", fontsize = 20)
        else:
            ax.set_xticklabels([])
            ax.set_xlabel("")

        ax.set_ylabel(r2.replace("_", "\n").title(), fontsize = 20)
        ax.set_yticklabels(ax.get_yticklabels(), fontsize = 14)
        plt.suptitle(f"{variant}: {key}\n$r^2$ Measures", y = 1.09)
        ax.axhline(0, color = "k", ls = "-", linewidth = 2)
        # ax.set_xticklabels(ax.get_xticklabels(), rotation = 90)
    return fig, axs

from plotly.subplots import make_subplots
import plotly.graph_objects as go

def dict_of_figs_to_dropdown_fig(figs, show_fig = True):
    keys = figs.keys()
    num_figs = len(keys)
    num_traces = {key: len(fig.data) for key, fig in figs.items()}
    show_traces = {key: [False for t in range(sum(num_traces.values()))] for key in keys}
    start_trace_index = {key: sum(list(num_traces.values())[:i]) for i, key in enumerate(keys)}
    end_trace_index = {key: sum(list(num_traces.values())[:i]) for i, key in enumerate(keys)}
    for key in keys:
        show_traces[key][start_trace_index[key]:end_trace_index[key] + 1] = [True for t in range(num_traces[key])]
    # Initialize the combined figure
    num_rows = max([int(fig.data[-1].xaxis.replace("x", "")) for name, fig in figs.items()])
    print(num_rows)
    combined_fig = make_subplots(rows=num_rows, cols=1, shared_xaxes=True)
    for key, fig in figs.items():
        for trace in fig.data:
            combined_fig.add_trace(trace)
        # Link the layout from the figure to the combined fig
        # combined_fig.update_layout(fig.layout, overwrite=False)

        layout_json = fig.layout.to_plotly_json()   
        # # Remove layout settings for rows that do not exist in the current figure
        current_num_rows = len(fig.data)
        # reset_layout = {}
        for i in range(current_num_rows + 1, num_rows + 1):

            for attr in ['yaxis', 'xaxis']:
                fig.layout[f'{attr}{i}'] = {}



    # Define buttons for the dropdown menu
        # Define buttons for the dropdown menu
    dropdown_buttons = [
        {
            "label": key,
            "method": "update",
            "args": [
                {"visible": show_traces[key]},
                {**figs[key].layout.to_plotly_json()}
            ]
        } for key in keys
    ]
    combined_fig.layout = {**figs[list(figs.keys())[0]].layout.to_plotly_json()}
    # Add dropdown menu to the figure layout
    combined_fig.update_layout(
            updatemenus=[
                {
                    "buttons": dropdown_buttons,
                    "direction": "down",
                    "showactive": True,
                    "x": 0.5,  # Center the dropdown menu horizontally
                    "xanchor": "center",  # Set the anchor to the center
                    "y": 1.15,  # Position above the plot area
                    "yanchor": "top"  # Set the anchor to the top
                }
            ]
        )

    combined_fig.update_traces(visible=False)
    for i, trace in enumerate(combined_fig.data):
        trace.visible = show_traces[list(keys)[0]][i]
    # Show combined figure
    if show_fig == True:
        combined_fig.show()
        
    return combined_fig


