import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from homebrewedFunctions.functions import *

data_cats = ["NoTaxRatios","WithTaxRatios"]
dfs_dct = {}
for data_cat in data_cats:
    filename = f"panelData{data_cat}.xlsx"
    dfs_dct[data_cat] = pd.read_excel(filename,sheet_name=None)
    for key, df in dfs_dct[data_cat].items():
        dfs_dct[data_cat][key].set_index(["State","DATE"], inplace = True)
        del dfs_dct[data_cat][key]["Unnamed: 0"]
plt.rcParams.update({'font.size': 12})
colors = ["EFNA", "Average 3-6"]
plot_keys = ["Total Income Taxes", "GDP", "Population", "GDP Per Capita",
             "Total Taxes", "Property Tax (T01)", 
         #    "Individual Income Tax (T40)", "Corp Net Income Tax (T41)"
            ] + colors

# Example usage
for variant, dct in list(dfs_dct.items())[:]: 
    for name, df in list(dct.items())[1:]:
    # possible to include tax component levels instead of component by percent of total taxes; 
    #  results of logged components in the least not meaningful without level info
        c ="Average 3-6"
        plot_df = df[plot_keys].dropna()
        fig, ax = plot_scatter_corr(plot_df.dropna(), f"{variant}: {name}\nColor: {c}", corr="corr", alpha =.25, cmap = "coolwarm", c = c)
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
from linearmodels import PanelOLS
from linearmodels.panel import compare

# Define plotting functions using Plotly
def plot_scatter_corr(data, x, y, title):
    fig = px.scatter(data, x=x, y=y, trendline='ols', opacity=0.25, color_continuous_scale='coolwarm')
    fig.update_layout(title=title)
    return fig

def plot_r2(data, title):
    r2_names = ["rsquared", "rsquared_within", "rsquared_between"]
    rows = len(r2_names)
    colors = {}
    for i, key in enumerate(data["y"].unique()):
        colors[key] = f'rgba({i*30 % 256}, {i*60 % 256}, {i*90 % 256}, 0.8)'
    fig = make_subplots(rows=rows, cols=1, subplot_titles=r2_names)   
    for j, r2 in enumerate(r2_names):
        print(r2)
        row = j+1
        df = data[data["r2"]==r2]
        showlegend = True if row == 1 else False 
        for i, yname in enumerate(df["y"]):
            print(df.iloc[i])            
            fig.add_trace(go.Bar(name=yname, x=df.keys()[2:], y=df.iloc[i].iloc[2:],
                          marker_color=colors[yname], legendgroup = yname, 
                          showlegend = showlegend), row=row, col=1)        
    fig.update_layout(barmode='group', title=title, showlegend=True)
    return fig

def compare_regs_plot(compare_regs, plot_params, title):
    rows = len(plot_params.keys())
    fig = make_subplots(rows=rows, cols=1, subplot_titles=list(plot_params.values()))

    # Define colors for the bars
    colors = {}
    for i, key in enumerate(compare_regs.params.index):
        colors[key] = f'rgba({i*30 % 256}, {i*60 % 256}, {i*90 % 256}, 0.8)'

    for key in compare_regs.params.index:
        for i, (param_method, param_name) in enumerate(plot_params.items()):
            row = i+1
            showlegend = True if row == 1 else False 
            fig.add_trace(go.Bar(name=key, x=compare_regs.params.columns, y=getattr(compare_regs, param_method).loc[key],
                                 marker_color=colors[key], legendgroup = key, showlegend = showlegend), row=row, col=1)
        # fig.add_trace(go.Bar(name=key, x=compare_regs.params.columns, y=compare_regs.pvalues.loc[key],
        #                      marker_color=colors[key], legendgroup = key, showlegend = False), row=2, col=1)

    fig.update_layout(barmode='group', title=title, showlegend=True)

    return fig

# Extract variables and models from the existing script context
dependent_variables = ["Total Taxes", "GDP", "State and Local Spending", "Property Tax (T01)"]
reg_vars = ["GDP", "Population", "Total Taxes", "Home Price Index", "Average 3-6", "Property Tax (T01)", "State and Local Spending"]
models = list(dfs_dct.keys())

plt.rcParams.update({'font.size': 32})

plot_params  = {'params':"Parameter Values",
 'pvalues': "P-values",
 'std_errors':"Standard Errors",
 'tstats':"T-statistics"}


for variant, dct in list(dfs_dct.items())[:]:
    r2_dict = {}
    results_dict = {}
    beta_dict = {}
    pval_dict = {}
    for key, df in list(dct.items())[1:]:
        results_dict[key] = {}
        r2_dict[key] = {}
        beta_dict[key] = {}
        pval_dict[key] = {}
        for y_name in dependent_variables:
            r2_dict[key][y_name] = {}
            results_dict[key][y_name] = {}
            beta_dict[key][y_name] = {}
            pval_dict[key][y_name] = {}
            X_names = [name for name in reg_vars if name != y_name]
            for i, entity in enumerate([False, True]):
                for j, time in enumerate([False, True]):
                    reg_data = df[reg_vars].dropna()
                    Y = reg_data[[y_name]]
                    X = reg_data[X_names]
                    if "Diff" not in key:
                        X["Constant"] = 1
                    model = PanelOLS(Y, X, entity_effects=entity, time_effects=time)
                    results_dict[key][y_name][f"Entity:{entity},\nTime:{time}"] = model.fit(cov_type='clustered', cluster_entity=True)
                    results = results_dict[key][y_name][f"Entity:{entity},\nTime:{time}"]
                    r2_dict[key][y_name][f"Entity:{entity},\nTime:{time}"] = {
                        'rsquared': results.rsquared,
                        'rsquared_between': results.rsquared_between,
                        'rsquared_within': results.rsquared_within
                    }
                    beta_dict[key][y_name][f"Entity:{entity},\nTime:{time}"] = results.params
                    pval_dict[key][y_name][f"Entity:{entity},\nTime:{time}"] = results.pvalues
            compare_regs = compare(results_dict[key][y_name])
            compfig = compare_regs_plot(compare_regs, plot_params, title=f"y = {y_name}\n{key}")
            compfig.write_html(f"outputs/{key}{y_name}{variant}Params.html")

        data_tuples = [(outer_key, inner_key, k, v) for outer_key, inner_dict in r2_dict[key].items() for inner_key, inner_inner_dict in inner_dict.items() for k, v in inner_inner_dict.items()]
        r2_df = pd.DataFrame(data_tuples, columns=['y', 'Effects', 'r2', 'Value']).sort_values(["r2", "y"])
        r2_df = r2_df.pivot(index=["r2", "y"], columns="Effects", values="Value").reset_index()
        r2fig = plot_r2(r2_df, f"{variant}: {key}\nR2 Values")
        r2fig.write_html(f"outputs/{key}{variant}R2s.html")

