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
dfs_dct