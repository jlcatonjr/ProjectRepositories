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

plt.rcParams.update({'font.size': 32})
r2_dict = {}
results_dict = {}
beta_dict = {}
pval_dict = {}

reg_vars = ["GDP", "Population", "Total Taxes", "Home Price Index",
            "Average 3-6", "Property Tax (T01)", "State and Local Spending"]
ic = info_criterion()
# possible to include tax component levels instead of component by percent of total taxes; 
#  results of logged components in the least not meaningful without level info
for variant, dct in list(dfs_dct.items())[:]:
    for key, df in list(dct.items())[1:]:
        results_dict[key] = {}
        r2_dict[key] = {}
        beta_dict[key] = {}
        pval_dict[key] = {}
        # fig, ax = plt.subplots(2,2, figsize = (20,20))
        for y_name in ["Total Taxes", "GDP", "State and Local Spending", "Property Tax (T01)"]:
            r2_dict[key][y_name] = {}
            results_dict[key][y_name] = {}
            beta_dict[key][y_name] = {}
            pval_dict[key][y_name] = {}
            X_names = [name for name in reg_vars if name != y_name]
            for i, entity in enumerate([False, True]):
                for j, time in enumerate([False, True]):
                    beta_dict[key][y_name] = {}
                    pval_dict[key][y_name] = {}
                    reg_data = df[reg_vars].dropna()
                    Y = reg_data[[y_name]]
                    X = reg_data[X_names]
                    if "Diff" not in key:
                        X["Constant"] = 1
                    n = reg_data.shape[1]
                    k = len(X)
                    # call panel_regression method
                    model = PanelOLS(Y,X, entity_effects=entity, time_effects=time)
                    # print(f"Data: {key} y={y_name}\n Entity: {entity}\nTime: {time}")
                    results_dict[key][y_name][f"Entity:{entity},\nTime:{time}"]  = model.fit(cov_type='clustered', cluster_entity=True)
                    results = results_dict[key][y_name][f"Entity:{entity},\nTime:{time}"]
                    r2_dict[key][y_name][f"Entity:{entity},\nTime:{time}"] = {}
                    r2s = ('rsquared', 'rsquared_between', 'rsquared_within')
                    for r2 in r2s:
                        r2_dict[key][y_name][f"Entity:{entity},\nTime:{time}"][r2] = getattr(results, r2)
                    # print(results)
            print(f"{variant}: {key}\n",y_name)
            compare_regs = compare(results_dict[key][y_name])
            print(compare_regs)
            compare_regs_plot(compare_regs, y_name, variant = variant, title = f"y = {y_name}\n{key}")
        # for r2 in r2s:
        #     print(r2df)
        #     r2_index = r2df.loc[r2] >0
    
        #     r2df = r2df[r2df.loc[r2] >0]
        # Convert the dictionary to a list of tuples
        data_tuples = [(outer_key, inner_key, k, v) for outer_key, inner_dict in r2_dict[key].items() for inner_key, inner_inner_dict in inner_dict.items() for k, v in inner_inner_dict.items()]
        
        # Convert the list of tuples to a pandas DataFrame
        r2_df = pd.DataFrame(data_tuples, columns=[ 'y', 'Effects', 'r2',  'Value']).sort_values(["r2", "y"])
        
        # Set the multi-level index
        r2_df = r2_df.pivot(index = ["r2", "y"], columns = "Effects", values = "Value").reset_index()
        plot_r2(r2_df, r2s, key,variant)