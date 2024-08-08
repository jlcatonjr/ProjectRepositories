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
check_path = "../outputs/EFNA"
check_dir = os.listdir(check_path)
for file in check_dir:
    files_to_copy.append(f"{check_path}/{file}")
files_to_copy.append("../outputs/PersonalIncome/LinePlotsStatePI.html")
for kind in ("Level", "Percent of Personal Income", "Real Level", "Real Value Per Capita"):
    files_to_copy.append(f"../outputs/PersonalIncome/ScatterPlotsPersonalIncome{kind}.html")
    # for kind in ("Percent of Expenditure", "Percent of General Revenue", "Percent of GDP", "Value Per Capita"):
    #     files_to_copy.append(f"../outputs/{state} government amount/AreaPlots{state} government amount{kind}TaxesFigs.html")
    #     files_to_copy.append(f"../outputs/{state} government amount/AreaPlots{state} government amount{kind}Revenue Source by GovernmentFigs.html")
    #     files_to_copy.append(f"../outputs/{state} government amount/MapPlotsByVariableAndYear{state} government amount{kind}.html")
    #     files_to_copy.append(f"../outputs/{state} government amount/LinePlotsStateFinancesAsPercentRevenuePercentGDPAndPerCapitaFigs{state} government amount.html")
for file in files_to_copy:    
    copyfile(file, "InteractiveHTML/"+file.split("/")[-1])

os.system("jupyter nbconvert --to html ImportTemplates.ipynb")