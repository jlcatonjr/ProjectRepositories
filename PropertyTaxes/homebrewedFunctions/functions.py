import time
import numpy as np
import pandas as pd
import pandas_datareader.data as web
import pingouin
import matplotlib.pyplot as plt
import datetime
from linearmodels import PanelOLS
from linearmodels.panel import compare
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px
import plotly.colors as colors

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
    # num_rows = max([int(fig.data[-1].xaxis.replace("x", "")) for name, fig in figs.items()])
    num_rows = max([len([i for i in fig.select_yaxes()]) for k, fig in figs.items()])
    print(num_rows)
    combined_fig = make_subplots(rows=num_rows, cols=1, shared_xaxes=True)
    for key, fig in figs.items():
        for trace in fig.data:
            combined_fig.add_trace(trace)
        # Link the layout from the figure to the combined fig
        # combined_fig.update_layout(fig.layout, overwrite=False)

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



import plotly.graph_objs as go
from plotly.subplots import make_subplots
import plotly.colors as colors

def create_scatter_dropdown(df, filename="interactive_scatter_plot.html", show_fig=False):
    # Create a subplot with 1 row and 1 column
    fig = make_subplots(rows=1, cols=1)
    scatter = go.Scatter(
        x=df[df.columns[0]],
        y=df[df.columns[0]],
        mode='markers',
        marker=dict(color=df[df.columns[0]],
                    colorscale='Viridis', size=14, opacity=0.3, colorbar=dict(thickness=20, title=dict(text=df.columns[0])))
    )

    fig.add_trace(scatter, row=1, col=1)
    fig.update_layout(
        xaxis_title=df.columns[0],
        yaxis_title=df.columns[0],
    )

    initial_hovertemplate = f"%{{xaxis.title.text}}: %{{x}}<br>%{{yaxis.title.text}}: %{{y}}<br>{df.columns[0]}: %{{marker.color}}"
    fig.update_traces(hovertemplate=initial_hovertemplate)

    # Extract unique states and years from the DataFrame
    states = df.index.get_level_values("State").unique()
    years = df.index.get_level_values("Year").unique()

    # Function to update marker opacity based on state, year, and slider value
    def update_opacity_and_size(selected_state=None, selected_year=None, opacity_slider_value=0.1, size_slider_value=10):
        opacities = [opacity_slider_value] * len(df)
        sizes = [size_slider_value] * len(df)
        if selected_state:
            state_mask = df.index.get_level_values("State") == selected_state
            opacities = [1 if mask else opacity for mask, opacity in zip(state_mask, opacities)]
            sizes = [size + 7 if mask else size for mask, size in zip(state_mask, sizes)]

        if selected_year:
            year_mask = df.index.get_level_values("Year") == str(selected_year)
            opacities = [1 if mask else opacity for mask, opacity in zip(year_mask, opacities)]
            sizes = [size + 5 if mask else size for mask, size in zip(year_mask, sizes)]
        return opacities, sizes

    # Create dropdown menus for x, y, color, and colorscale
    x_buttons = [dict(args=[{"x": [df[col]]}, {"xaxis.title.text": col}], label=col, method="update") for col in df.columns]
    y_buttons = [dict(args=[{"y": [df[col]]}, {"yaxis.title.text": col}], label=col, method="update") for col in df.columns]
    color_buttons = [dict(args=[{"marker.color": [df[col]], 
                                 "marker.colorbar.title.text": col,
                                "hovertemplate":f"%{{xaxis.title.text}}: %{{x}}<br>%{{yaxis.title.text}}: %{{y}}<br>{col}: %{{marker.color}}" }], label=col, method="update") for col in df.columns]
    color_scales = colors.PLOTLY_SCALES.keys()
    colorscale_buttons = [dict(args=[{"marker.colorscale": scale}], label=scale, method="update") for scale in color_scales]

    # Create dropdown menus for state and year selection
    state_buttons = [dict(args=[{"marker.opacity": [update_opacity_and_size(selected_state=state)[0]],
                                 "marker.size": [update_opacity_and_size(selected_state=state)[1]]
                                 }], 
                          label=state, method="update") for state in states]
    year_buttons = [dict(args=[{"marker.opacity": [update_opacity_and_size(selected_year=year)[0]],
                                 "marker.size": [update_opacity_and_size(selected_year=year)[1]]
                                 }], 
                         label=str(year), method="update") for year in years]

    sliders = [{
                "active": 0,
                "currentvalue": {"prefix": "Marker Size: "},
                "pad": {"t": 50},
                "steps": [
                    {"label": str(size), "method": "restyle", "args": ["marker.size", [size]]}
                    for size in range(1, 31)
                ],
                "x": 0, "len": .5, "xanchor": "left", "y": 0
            },
            {
                "active": 7,
                "currentvalue": {"prefix": "Marker Opacity: "},
                "pad": {"t": 50},
                "steps": [
                    {"label": str(opacity), "method": "restyle", "args": ["marker.opacity", [opacity]]}
                    for opacity in [round(x * 0.01, 2) for x in range(1, 101)]
                ],
                "x": 0.5, "len": .5, "xanchor": "left", "y": 0
            }
        ]

    fig.update_layout(
        updatemenus=[
            dict(buttons=x_buttons, direction="down", showactive=True, x=0.17, xanchor="left", y=1.15, yanchor="top"),
            dict(buttons=y_buttons, direction="down", showactive=True, x=0.32, xanchor="left", y=1.15, yanchor="top"),
            dict(buttons=color_buttons, direction="down", showactive=True, x=0.47, xanchor="left", y=1.15, yanchor="top"),
            dict(buttons=colorscale_buttons, direction="down", showactive=True, x=0.62, xanchor="left", y=1.15, yanchor="top"),
            dict(buttons=state_buttons, direction="down", showactive=True, x=0.77, xanchor="left", y=1.15, yanchor="top"),
            dict(buttons=year_buttons, direction="down", showactive=True, x=0.92, xanchor="left", y=1.15, yanchor="top")
        ],
        sliders=sliders,
        annotations=[
            dict(text="X-axis", x=0.17, xref="paper", y=1.25, yref="paper", xanchor="left", showarrow=False),
            dict(text="Y-axis", x=0.32, xref="paper", y=1.25, yref="paper", xanchor="left", showarrow=False),
            dict(text="Color", x=0.47, xref="paper", y=1.25, yref="paper", xanchor="left", showarrow=False),
            dict(text="Colorscale", x=0.62, xref="paper", y=1.25, yref="paper", xanchor="left", showarrow=False),
            dict(text="State", x=0.77, xref="paper", y=1.25, yref="paper", xanchor="left", showarrow=False),
            dict(text="Year", x=0.92, xref="paper", y=1.25, yref="paper", xanchor="left", showarrow=False)
        ]
    )

    # Custom JavaScript to handle colorbar title
    custom_js = '''
    <script>
        const colorMenu = document.querySelectorAll('[data-title="Color"]')[0];
        colorMenu.addEventListener('click', function(event) {
            const target = event.target;
            if (target.tagName === 'g' || target.tagName === 'text') {
                const newTitle = target.textContent;
                Plotly.relayout('plot', {'marker.colorbar.title.text': newTitle});
            }
        });
    </script>
    '''

    # Add the custom JS to the HTML output
    with open(filename, 'w') as f:
        f.write(fig.to_html(full_html=False, include_plotlyjs='cdn'))
        f.write(custom_js)

    if show_fig:
        fig.show()

        


    # fig.write_html(filename)


# import plotly.graph_objects as go
# from plotly.subplots import make_subplots

# def dict_of_figs_to_dropdown_fig(figs, show_fig=True):
#     keys = figs.keys()
#     num_figs = len(keys)
#     num_traces = {key: len(fig.data) for key, fig in figs.items()}
#     show_traces = {key: [False for t in range(sum(num_traces.values()))] for key in keys}
#     start_trace_index = {key: sum(list(num_traces.values())[:i]) for i, key in enumerate(keys)}
#     end_trace_index = {key: sum(list(num_traces.values())[:i]) for i, key in enumerate(keys)}
    
#     for key in keys:
#         show_traces[key][start_trace_index[key]:end_trace_index[key] + 1] = [True for t in range(num_traces[key])]

#     num_rows = max([len([i for i in fig.select_yaxes()]) for k, fig in figs.items()])
#     combined_fig = make_subplots(rows=num_rows, cols=1, shared_xaxes=True)

#     for key, fig in figs.items():
#         for trace in fig.data:
#             combined_fig.add_trace(trace)

#     dropdown_buttons_keys = [
#         {
#             "label": key,
#             "method": "update",
#             "args": [
#                 {"visible": show_traces[key]},
#                 {**figs[key].layout.to_plotly_json()}
#             ]
#         } for key in keys
#     ]

#     combined_fig.update_layout(
#         updatemenus=[
#             {
#                 "buttons": dropdown_buttons_keys,
#                 "direction": "down",
#                 "showactive": True,
#                 "x": 0.5,
#                 "xanchor": "center",
#                 "y": 1.15,
#                 "yanchor": "top"
#             }
#         ]
#     )
#     # combined_fig.layout = {**figs[list(figs.keys())[0]].layout.to_plotly_json()}

#     combined_fig.update_traces(visible=False)
#     for i, trace in enumerate(combined_fig.data):
#         trace.visible = show_traces[list(keys)[0]][i]

#     if show_fig:
#         combined_fig.show()

#     # Add the original menus from each figure to the combined figure
#     # for fig in figs.values():
#     #     if 'updatemenus' in fig.layout:
#     #         for menu in fig.layout.updatemenus:
#     #             combined_fig.update_layout(
#     #                 updatemenus=[*combined_fig.layout.updatemenus, menu]
#     #             )

#     return combined_fig
