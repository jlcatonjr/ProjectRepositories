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
from pandas.api.types import is_numeric_dtype

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


def line_dropdown(df, regions_df, title = ""):
    y0_name = list(df.keys())[0]
    menu_font =dict(size=20)

    plot_df = df.reset_index()
    fig = px.line(plot_df.dropna(subset = y0_name,axis = 0), x="Year", y=y0_name, color="State", title = title)
    initial_hovertemplate = f"%{{x}}<br>%{{yaxis.title.text}}: %{{y}}"
    fig.update_traces(hovertemplate=initial_hovertemplate)

    y_buttons = []
    for col in df.columns:
        if is_numeric_dtype(df[col]):
            y_buttons.append(
                dict(
                    args=[
                        {"y": [df.loc[state][col].dropna(axis = 0) for state in plot_df['State'].unique()]},
                        {"yaxis.title.text": col}
                    ],
                    label=col,
                    method="update"
                )
            )

    regdiv_buttons = {"Region": [], 
                "Division": []}
    # button to show all states
    regdiv_buttons["Region"].append(dict(
                args=[{"visible": df.index.get_level_values("State").unique().isin(df.index.get_level_values("State").unique()),
                        "font":menu_font}],
                label="All",
                method="update",
                ))
    for regdiv_key in regdiv_buttons:
        
        regions = regions_df[regdiv_key].unique()
        for region in regions:
            states_in_region = regions_df[regions_df[regdiv_key] == region]['State Abbrev'].values
            visible_states = df.index.get_level_values("State").unique().isin(states_in_region)
            regdiv_buttons[regdiv_key].append(
                dict(
                    args=[{"visible": visible_states,
                           "font":menu_font}],
                    label=region,
                    method="update"
                )
            )
        
    fig.update_layout(
        updatemenus=[
            dict(
                buttons=y_buttons,
                direction="down",
                showactive=True,
                x=0,
                xanchor="left",
                y=1.3,
                yanchor="top",
            ),
            dict(
                type="buttons",
                direction="left",
                buttons=[
                    dict(
                        args=[{"yaxis.type": "linear",
                               "font":menu_font}],
                        label="Linear Y",
                        method="relayout"
                    ),
                    dict(
                        args=[{"yaxis.type": "log",
                               "font":menu_font}],
                        label="Log Y",
                        method="relayout"
                    )
                ],
                x=0,
                xanchor="left",
                y=1.24,
                yanchor="top"
            ),
            dict(
                type="buttons",
                direction="left",
                buttons=regdiv_buttons["Region"],
                x=0,
                xanchor="left",
                y=1.18,
                yanchor="top",
            ),
          
                dict(
                type="buttons",
                direction="left",
                buttons=regdiv_buttons["Division"][:len(regdiv_buttons["Division"])//2+1],
                x=0,
                xanchor="left",
                y=1.12,
                yanchor="top",
            ),
            dict(
                type="buttons",
                direction="left",
                buttons=regdiv_buttons["Division"][len(regdiv_buttons["Division"])//2+1:],
                x=0,
                xanchor="left",
                y=1.06,
                yanchor="top",
            )
        ],
    )
    fig.update_layout(
        margin=dict(t=200),
        font=dict(size=20),
        clickmode='event+select',
        hovermode='closest',    
        template='plotly_white',
        updatemenus=[dict(font=dict(size=20), yanchor='top')],
        # autosize=True
        # dragmode=False
    )

    return fig




def create_scatter_dropdown(df, regions_df, title = "",
                            filename="interactive_scatter_plot.html", 
                            entity = "State", time = "Year", show_fig=False):
    # Create a subplot with 1 row and 1 column
    fig = make_subplots(rows=1, cols=1)
    scatter = go.Scatter(
        x=df[df.columns[0]],
        y=df[df.columns[0]],
        mode='markers',
        marker=dict(color=df[df.columns[0]],
                    colorscale='Viridis', size=14, opacity=0.3, colorbar=dict(thickness=20, title=dict(text=df.columns[0]))),
        text = df.index.get_level_values(entity) + ": " + df.index.get_level_values(time),
    )

    fig.add_trace(scatter, row=1, col=1)
    fig.update_layout(
        xaxis_title=df.columns[0],
        yaxis_title=df.columns[0],
        title = dict(text=title, x=1, xanchor='right')
        )

    initial_hovertemplate = f"%{{text}}<br>%{{xaxis.title.text}}: %{{x}}<br>%{{yaxis.title.text}}: %{{y}}<br>{df.columns[0]}: %{{marker.color}}"
    fig.update_traces(hovertemplate=initial_hovertemplate)

    # Extract unique states and years from the DataFrame
    controls = {c: df.index.get_level_values(c).unique() for c in [entity, time]}


    # Function to update marker opacity based on state, year, and slider value
    def update_opacity_and_size(selected_states=None, selected_year=None, opacity_slider_value=0.1,
                                 size_slider_value=10):
        opacities = [opacity_slider_value] * len(df)
        sizes = [size_slider_value] * len(df)
        if selected_states is not None:
            state_mask = df.index.get_level_values(entity).isin(selected_states)
            opacities = [1 if mask else opacity for mask, opacity in zip(state_mask, opacities)]
            sizes = [size + 7 if mask else size for mask, size in zip(state_mask, sizes)]

        if selected_year != None:
            year_mask = df.index.get_level_values(time) == str(selected_year)
            opacities = [1 if mask else opacity for mask, opacity in zip(year_mask, opacities)]
            sizes = [size + 5 if mask else size for mask, size in zip(year_mask, sizes)]
        return opacities, sizes

    # Create dropdown menus for x, y, color, and colorscale
    x_buttons = [dict(args=[{"x": [df[col]]}, {"xaxis.title.text": col}], label=col, method="update") for col in df.columns]
    y_buttons = [dict(args=[{"y": [df[col]]}, {"yaxis.title.text": col}], label=col, method="update") for col in df.columns]
    color_buttons = [dict(args=[{"marker.color": [df[col]], 
                                 "marker.colorbar.title.text": col,
                                "hovertemplate":f"%{{text}}<br>%{{xaxis.title.text}}: %{{x}}<br>%{{yaxis.title.text}}: %{{y}}<br>{col}: %{{marker.color}}" }], label=col, method="update") for col in df.columns]
    color_scales = colors.PLOTLY_SCALES.keys()
    colorscale_buttons = [dict(args=[{"marker.colorscale": scale}], label=scale, method="update") for scale in color_scales]
    
    ## Create Region Buttons
    regdiv_buttons = {"Region":[],
                       "Division":[]}
    for i, regdiv_key in enumerate(regdiv_buttons):
        regions = regions_df[regdiv_key].unique()
        for region in regions:
            regdiv_buttons[regdiv_key].append(
                {"label": str(region), 
                 "method": "update", 
                 "args": [{
                    "marker.opacity": [update_opacity_and_size(selected_states=regions_df[regions_df[regdiv_key] == region]['State Abbrev'].values)[0]],
                    "marker.size": [update_opacity_and_size(selected_states=regions_df[regions_df[regdiv_key] == region]['State Abbrev'].values)[1]]}
                    ]}
            )

    size_slider = {
                    "active": 0, "currentvalue": {"prefix": "Marker Size: "}, "pad": {"t": 50},
                    "steps": [
                        {"label": str(size), "method": "restyle", "args": ["marker.size", [size]]}
                        for size in range(1, 31)
                    ],
                    "x": 0, "len": .5, "xanchor": "left", "y": -0.3
                }
    opacity_slider = {
                    "active": 7, "currentvalue": {"prefix": "Marker Opacity: "}, "pad": {"t": 50},
                    "steps": [
                        {"label": str(opacity), "method": "restyle", "args": ["marker.opacity", [opacity]]}
                        for opacity in [round(x * 0.01, 2) for x in range(1, 101)]
                    ],
                    "x": 0.5, "len": .5, "xanchor": "left", "y": -0.3
                }
    year_slider = {
                "active": 0, "currentvalue": {"prefix": "Year: "}, "pad": {"t": 50},
                "steps": [
                    {"label": str(year), "method": "update", "args": [{"marker.opacity": [update_opacity_and_size(selected_year=year)[0]],
                                                                    "marker.size": [update_opacity_and_size(selected_year=year)[1]]}]}
                    for year in controls[time]
                ],
                "x": 0, "len": 1.0, "xanchor": "left", "y": 0
            }
    state_slider = {
                "active": 0, "currentvalue": {"prefix": "State: "}, "pad": {"t": 50},
                "steps": [
                    {"label": str(ent), "method": "update", "args": [{"marker.opacity": [update_opacity_and_size(selected_states=[ent])[0]],
                                                                    "marker.size": [update_opacity_and_size(selected_states=[ent])[1]]}]}
                    for ent in controls[entity]
                ],
                "x": 0, "len": 1.0, "xanchor": "left", "y": -0.15
            }




    sliders = [size_slider, opacity_slider, year_slider, state_slider]
    annotation_y = 1.125
    fig.update_layout(
        updatemenus=[
            dict(buttons=x_buttons, direction="down", showactive=True, x=0, xanchor="left", y=1.085, yanchor="top"),
            dict(buttons=y_buttons, direction="down", showactive=True, x=0.25, xanchor="left", y=1.085, yanchor="top"),
            dict(buttons=color_buttons, direction="down", showactive=True, x=0.5, xanchor="left", y=1.085, yanchor="top"),
            dict(buttons=colorscale_buttons, direction="down", showactive=True, x=0.75, xanchor="left", y=1.085, yanchor="top"),
            dict(type="buttons", direction="left", buttons=regdiv_buttons["Region"], x=0, xanchor="left", y=1.37, yanchor="top"),         
            dict(type="buttons", direction="left", buttons=regdiv_buttons["Division"][:len(regdiv_buttons["Division"])//2+1],
               x=0, xanchor="left", y=1.27, yanchor="top"),
            dict(type="buttons", direction="left", buttons=regdiv_buttons["Division"][len(regdiv_buttons["Division"])//2+1:],
                x=0, xanchor="left", y=1.2, yanchor="top"
            )            # dict(buttons=state_buttons, direction="down", showactive=True, x=0.77, xanchor="left", y=1.15, yanchor="top"),
            # dict(buttons=year_buttons, direction="down", showactive=True, x=0.92, xanchor="left", y=1.15, yanchor="top"),
            # dict(type="buttons", direction="left", buttons=regdiv_buttons["Region"], x=0.38, xanchor="left", y=1.15, yanchor="top"),
            # dict(type="buttons", direction="left", buttons=regdiv_buttons["Division"], x=0.38, xanchor="left", y=1.05, yanchor="top")
        ],
        sliders=sliders,
        annotations=[
            dict(text="X-axis", x=0, xref="paper", y=annotation_y, yref="paper", xanchor="left", showarrow=False),
            dict(text="Y-axis", x=0.25, xref="paper", y=annotation_y, yref="paper", xanchor="left", showarrow=False),
            dict(text="Color", x=0.5, xref="paper", y=annotation_y, yref="paper", xanchor="left", showarrow=False),
            dict(text="Colorscale", x=0.75, xref="paper", y=annotation_y, yref="paper", xanchor="left", showarrow=False),
            dict(text="Region", x=-0.005, xref="paper", y=1.35, yref="paper", xanchor="right", showarrow=False),
            dict(text="Division", x=-0.005, xref="paper", y=1.22, yref="paper", xanchor="right", showarrow=False),
            # dict(text="State", x=0.77, xref="paper", y=1.25, yref="paper", xanchor="left", showarrow=False),
            # dict(text="Year", x=0.92, xref="paper", y=1.25, yref="paper", xanchor="left", showarrow=False)
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

    # enhance_plotly_figure_for_mobile(fig, output_path=filename)
    with open(filename, 'w') as f:
        f.write(fig.to_html(full_html=False, include_plotlyjs='cdn'))
        f.write(custom_js)
        
    # enhance_plotly_html_for_mobile(filename, output_path=filename)
    if show_fig:
        fig.show()

        
    # return fig

    # fig.write_html(filename)


import plotly.express as px
import pandas as pd

def create_map(df, name, title ="", stable_cbar = True, entity_name = "State", time_name = "DATE"):
    plot_df = df[[entity_name, time_name, name]].dropna()
    dates = sorted([str(d)[:4] for d in plot_df[time_name].unique()])
    plot_df[time_name] = plot_df[time_name].astype(str).str[:4]
    plot_df = plot_df.reset_index().pivot(index=[entity_name], columns=time_name, values=name)
    plot_df = plot_df.loc[plot_df.index != "US"].reset_index()

    init_var = dates[-1]

    # Calculate the overall min and max values across all years
    if stable_cbar:
        min_val = plot_df.drop(columns=entity_name).min().min()
        max_val = plot_df.drop(columns=entity_name).max().max()
    else:
        min_val = None
        max_val = None
    # Create the initial plot
    fig = px.choropleth(plot_df,
                        locations=entity_name,
                        color=init_var,  # Initial variable
                        color_continuous_scale='spectral_r',
                        locationmode='USA-states',
                        scope='usa')
    # Update the coloraxis to match the overall min and max
    fig.update_traces(hovertemplate="%{location}: %{z}<extra></extra>", 
                    selector=dict(type='choropleth'))
    fig.update_layout(
        coloraxis_colorbar=dict(title=''),
        coloraxis=dict(cmin=min_val, cmax=max_val, colorscale='spectral_r')
        )
    color_scales = colors.PLOTLY_SCALES.keys()
    # Create a slider for colormaps
    colorscale_steps = [
        dict(
            method='relayout',
            label=scale,
            args=[{'coloraxis.colorscale': scale}]
        ) for scale in color_scales
    ]

    colormap_slider = [dict(
        active=0,
        currentvalue={"prefix": "Colormap: "},
        pad={"t": 50},
        steps=colorscale_steps,
        xanchor="left",
        yanchor="top",
    )]

    # Update layout with slider
    date_slider = [dict(
        active=len(dates)-1,
        currentvalue={"prefix": "Year: "},
        pad={"t": 50},
        steps=[
            dict(
                label=date,
                method="update",
                args=[
                    {"z": [plot_df[date]]},
                    {"title": dict(text= f"{title}<br>{name}: {date}", x=0.5)},
                    {"hovertemplate": "%{location}: " + plot_df[date].astype(str) + "<extra></extra>"}
                ]
            ) for date in dates
        ],
        yanchor = "bottom",
        y = -0.05
    ),

    ]
    sliders = date_slider + colormap_slider

    fig.update_layout(
        title = dict(text= f"{title}<br>{name}: {init_var}", x=0.5),
        sliders=sliders,
        coloraxis_colorbar=dict(title=''),
        coloraxis=dict(cmin=min_val, cmax=max_val, colorscale='spectral_r')
        )

    return fig

def combine_map_figs(figs, title = ""):
    # Combine all figures into one figure with a menu to select between them
    combined_fig = make_subplots(rows=1, cols=1)
    names = list(figs.keys())
    for name, fig in figs.items():
        for trace in fig.data:
            combined_fig.add_trace(trace)
    # Create dropdown menu to switch between figures
    dropdown_buttons = [
        dict(
            label=name,
            method="update",
            args=[{"visible": [True if name == selected_name else False for selected_name in names]},
                {**figs[name].layout.to_plotly_json()},
                {"title": f"{title}<br>{name}"}]
        ) for name in names
    ]
    combined_fig.layout = {**figs[list(figs.keys())[0]].layout.to_plotly_json()}
    combined_fig.update_layout(
        updatemenus=[dict(buttons=dropdown_buttons, direction="down", showactive=True, x =1, y = 1, yanchor = "bottom", xanchor = "right")]#,
        # title_text=f"{title}<br>{names[0]}", title_x=0.5
    )

    # Set visibility for initial state
    combined_fig.for_each_trace(lambda trace: trace.update(visible=False))
    combined_fig.data[0].update(visible=True)
    return combined_fig



def dict_of_figs_to_dropdown_fig(figs, show_fig=True, use_sliders=False):
    keys = list(figs.keys())
    num_figs = len(keys)
    num_traces = {key: len(fig.data) for key, fig in figs.items()}
    start_trace_index = {key: sum(list(num_traces.values())[:i]) for i, key in enumerate(keys)}
    
    show_traces = {key: [False] * sum(num_traces.values()) for key in keys}
    for key in keys:
        start_index = start_trace_index[key]
        end_index = start_index + num_traces[key]
        show_traces[key][start_index:end_index] = [True] * num_traces[key]

    combined_fig = make_subplots(rows=1, cols=1, shared_xaxes=True)

    for fig in figs.values():
        for trace in fig.data:
            combined_fig.add_trace(trace)

    if use_sliders:
        sliders = [dict(
            active=0,
            pad={"t": 50},
            steps=[
                dict(
                    label=key,
                    method="update",
                    args=[
                        {"visible": show_traces[key]},
                        {**figs[key].layout.to_plotly_json()}
                    ]
                ) for key in keys
            ],
            x = 0, len = 1.0, xanchor = "left", y = 0, yanchor = "top"
        )]
        combined_fig.update_layout(sliders=sliders)
    else:
        dropdown_buttons_keys = [
            {
                "label": key,
                "method": "update",
                "args": [
                    {"visible": show_traces[key]},
                    {**figs[key].layout.to_plotly_json()}
                ]
            } for key in keys
        ]

        combined_fig.update_layout(
            updatemenus=[
                {
                    "buttons": dropdown_buttons_keys,
                    "direction": "down",
                    "showactive": False,
                    "x": 0.5,
                    "xanchor": "center",
                    "y": 1.15,
                    "yanchor": "top"
                }
            ]
        )

    combined_fig.update_traces(visible=False)
    for i, trace in enumerate(combined_fig.data):
        trace.visible = show_traces[keys[0]][i]
    combined_fig.update_layout(**figs[keys[0]].layout.to_plotly_json())

    if show_fig:
        combined_fig.show()

    return combined_fig

# def dict_of_line_figs_to_dropdown_fig(figs, show_fig=True, use_sliders=False):
#     keys = list(figs.keys())
#     combined_fig = make_subplots(rows=1, cols=1, shared_xaxes=True)
#     # trace mapping required to avoid duplicating legend entries for plots linked to buttons
#     trace_mapping = {}
#     for key in keys:
#         for trace in figs[key].data:
#             if trace.name not in trace_mapping:
#                 trace_mapping[trace.name] = len(combined_fig.data)
#                 combined_fig.add_trace(trace)
#             else:
#                 continue
#                 # combined_fig.add_trace(go.Scatter(visible=False))

#     visibility = {key: [False] * len(combined_fig.data) for key in keys}
#     for key in keys:
#         for trace_name in [trace.name for trace in figs[key].data]:
#             visibility[key][trace_mapping[trace_name]] = True

#     if use_sliders:
#         sliders = [dict(
#             pad={"t": 50},
#             steps=[
#                 dict(
#                     label=key,
#                     method="update",
#                     args=[
#                         {"visible": visibility[key]},
#                         {**figs[key].layout.to_plotly_json()}
#                         # {"data": figs[key].data}
#                     ]
#                 ) for key in keys
#             ],
#             active = True,

#             # x = 0, len = 1.0, xanchor = "left", y = 0, yanchor = "top"
#         )]

#         combined_fig.update_layout(sliders=sliders)


#     else:
#         dropdown_buttons_keys = [
#             {
#                 "label": key,
#                 "method": "restyle",
#                 "args": [
#                     {"visible": visibility[key]},
#                     {**figs[key].layout.to_plotly_json()}
#                 ]
#             } for key in keys
#         ]

#         combined_fig.update_layout(
#             updatemenus=[
#                 {
#                     "buttons": dropdown_buttons_keys,
#                     "direction": "down",
#                     "showactive": False,
#                     "x": 0.5,
#                     "xanchor": "center",
#                     "y": 1.15,
#                     "yanchor": "top"
#                 }
#             ]
#         )


#     combined_fig.update_traces(visible=False)
#     for i, trace in enumerate(combined_fig.data):
#         trace.visible = visibility[keys[0]][i]
#     combined_fig.update_layout(**figs[keys[0]].layout.to_plotly_json())

#     if show_fig:
#         combined_fig.show()

#     return combined_fig


import plotly.express as px
from plotly.subplots import make_subplots
from pandas.api.types import is_numeric_dtype

def aggregated_line_dropdown(dfs, regions_df,title = ""):
    fig = make_subplots(rows=1, cols=1)
    menu_font =dict(size=20)
    # Extract keys and initialize the first plot
    keys = list(dfs.keys())
    first_key = keys[0]
    plot_df = dfs[first_key].reset_index()
    y0 = list(dfs[first_key].keys())[0]
    fig = px.line(plot_df, x="Year", y=y0, color="State")
    initial_hovertemplate = f"%{{x}}<br>%{{yaxis.title.text}}: %{{y}}"
    fig.update_traces(hovertemplate=initial_hovertemplate)


    def create_menus(df_key):
        df = dfs[df_key]
        plot_df = df.reset_index()

        # Create y_buttons for all numeric columns
        y_buttons = []
        for col in df.columns:
            if is_numeric_dtype(df[col]):
                y_buttons.append(
                    dict(
                        args=[
                            {"y": [df.loc[state][col] for state in plot_df['State'].unique()]},
                            {"yaxis.title.text": col,
                            "font":menu_font}
                        ],
                        label=col,
                        method="update",
                )
                )

        # Create buttons for Region and Division
        regdiv_buttons = {"Region": [], 
                    "Division": []}
        # button to show all states
        regdiv_buttons["Region"].append(dict(
                    args=[{"visible": df.index.get_level_values("State").unique().isin(df.index.get_level_values("State").unique()),
                           "font":menu_font}],
                    label="All",
                    method="update",
                    ))
        for regdiv_key in regdiv_buttons:
            regions = regions_df[regdiv_key].unique()
            for region in regions:
                states_in_region = regions_df[regions_df[regdiv_key] == region]['State Abbrev'].values
                visible_states = df.index.get_level_values("State").unique().isin(states_in_region)
                regdiv_buttons[regdiv_key].append(
                    dict(
                        args=[{"visible": visible_states,
                               "font":menu_font}],
                        label=region,
                        method="update",
                        
                    )
                )
        
        menus = [
            dict(
                buttons=y_buttons,
                direction="down",
                showactive=True,
                x=0,
                xanchor="left",
                y=1.3,
                yanchor="top",
            ),
            dict(
                type="buttons",
                direction="left",
                buttons=[
                    dict(
                        args=[{"yaxis.type": "linear",
                               "font":menu_font}],
                        label="Linear Y",
                        method="relayout"
                    ),
                    dict(
                        args=[{"yaxis.type": "log",
                               "font":menu_font}],
                        label="Log Y",
                        method="relayout"
                    )
                ],
                x=0,
                xanchor="left",
                y=1.24,
                yanchor="top",
            ),
            dict(
                type="buttons",
                direction="left",
                buttons=regdiv_buttons["Region"],
                x=0,
                xanchor="left",
                y=1.18,
                yanchor="top",
            ),
            dict(
                type="buttons",
                direction="left",
                buttons=regdiv_buttons["Division"][:len(regdiv_buttons["Division"])//2+1],
                x=0,
                xanchor="left",
                y=1.12,
                yanchor="top",
            ),
            dict(
                type="buttons",
                direction="left",
                buttons=regdiv_buttons["Division"][len(regdiv_buttons["Division"])//2+1:],
                x=0,
                xanchor="left",
                y=1.06,
                yanchor="top",
            ),
        ]
        return menus

    # Create slider steps for each dataframe
    sliders = [
        dict(
            active=True,
            currentvalue={"prefix": ""},
            pad={"t": 50},
            y = 0, x=0, xanchor="left", yanchor="top",
            steps=[
                dict(
                    args=[
                        {
                            
                            "x": [dfs[key].loc[state].index.get_level_values('Year') for state in dfs[key].index.get_level_values("State").unique()],
                            "y": [dfs[key].loc[state][y0] for state in dfs[key].index.get_level_values("State").unique()],

                        },
                        {
                            "updatemenus": create_menus(key),
                            "yaxis.title.text": y0,
                            "title" : dict(text=f"{title}<br>{key}", x=1, xanchor='right'),
                            "font":menu_font                           
                        }
                    ],                    label=key,
                    method="update"
                )
                for key in keys
            ]
        )
    ]

    fig.update_layout(
        sliders=sliders,
        updatemenus=create_menus(first_key),
        margin=dict(t=200),
        font=dict(size=20),
        clickmode='event+select',
        hovermode='closest',
        title = dict(text=f"{title}<br>Level", x=1, xanchor='right')         
    )

    # Add custom CSS to improve touch interaction
    fig.update_layout(
        template='plotly_white',
        updatemenus=[dict(font=dict(size=20), yanchor='top')],
        # autosize=True
        # dragmode=False
    )
 
    return fig



def enhance_plotly_html_for_mobile(html_path, output_path=None):
    import re
    
    # Read the content of the HTML file
    with open(html_path, 'r') as file:
        html_content = file.read()
    
    # Meta tag and CSS to insert
    meta_and_css = """
    <meta name="viewport" content="width=device-width, initial-scale=1, shrink-to-fit=no">
    <style>
        .plotly-graph-div {
            width: 100% !important;
            height: 100vh !important;  /* Use viewport height */
        }
        .plotly .modebar-group {
            display: flex;
            justify-content: center;
        }
        .plotly .modebar-btn, .plotly .dropdown-menu {
            width: auto !important;
        }
        @media (max-width: 600px) {
            .plotly-graph-div {
                height: calc(100vh - 100px) !important;  /* Adjust height for mobile view */
            }
        }
    </style>
    """
    
    # Find the position to insert the meta tag and CSS
    head_end = html_content.find("</head>")
    
    if head_end != -1:
        # Insert the meta tag and CSS
        enhanced_html_content = html_content[:head_end] + meta_and_css + html_content[head_end:]
    else:
        # If </head> not found, just append meta tag and CSS at the beginning
        enhanced_html_content = meta_and_css + html_content
    
    # If output path is not provided, overwrite the original file
    if not output_path:
        output_path = html_path
    
    # Save the enhanced HTML content to the output file
    with open(output_path, 'w') as file:
        file.write(enhanced_html_content)

