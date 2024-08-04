import os
from shutil import copyfile
areas = {"Revenue":{"Revenue Source by Government" : [],
                    "Taxes": []},
        "Expenditure": {
            "Expenditures":[],
            "Expenditure by Function": []},
}
print(os.getcwd())
files_to_copy = []
for state in ("State", "State & local"):
    for revexpdebt in ("Revenue", "Expenditure", "Debt"):
        files_to_copy.append(f"../outputs/{state} government amount/LinePlotsStateFinances{revexpdebt}{state} government amount.html")
        for kind in ("Percent of Expenditure", "Percent of General Revenue", "Percent of GDP", "Real Value Per Capita"):
            files_to_copy.append(f"../outputs/{state} government amount/ScatterPlots{revexpdebt}{state} government amount{kind}.html")
            files_to_copy.append(f"../outputs/{state} government amount/MapPlotsByVariableAndYear{revexpdebt}{state} government amount{kind}.html")
    for revexp in areas.keys():
        for focus in areas[revexp].keys():
            for kind in ("Percent of Expenditure", "Percent of General Revenue", "Percent of GDP", "Real Value Per Capita"):
                files_to_copy.append(f"../outputs/{state} government amount/AreaPlots{revexp}{state} government amount{kind}{focus}Figs.html")

    # for kind in ("Percent of Expenditure", "Percent of General Revenue", "Percent of GDP", "Value Per Capita"):
    #     files_to_copy.append(f"../outputs/{state} government amount/AreaPlots{state} government amount{kind}TaxesFigs.html")
    #     files_to_copy.append(f"../outputs/{state} government amount/AreaPlots{state} government amount{kind}Revenue Source by GovernmentFigs.html")
    #     files_to_copy.append(f"../outputs/{state} government amount/MapPlotsByVariableAndYear{state} government amount{kind}.html")
    #     files_to_copy.append(f"../outputs/{state} government amount/LinePlotsStateFinancesAsPercentRevenuePercentGDPAndPerCapitaFigs{state} government amount.html")
for file in files_to_copy:    
    copyfile(file, file.split("/")[-1])

os.system("jupyter nbconvert --to html ImportTemplates.ipynb")