import os
from shutil import copyfile

print(os.getcwd())
files_to_copy = ["../outputs/State & local government amount/AreaPlotsState & local government amountPercent of General RevenueTaxesFigs.html",
                 "../outputs/State & local government amount/AreaPlotsState & local government amountPercent of General RevenueRevenue Source by GovernmentFigs.html",
                 "../outputs/State & local government amount/AreaPlotsState & local government amountPercent of GDPTaxesFigs.html",
                 "../outputs/State & local government amount/AreaPlotsState & local government amountPercent of GDPRevenue Source by GovernmentFigs.html",
                 "../outputs/State & local government amount/MapPlotsByVariableAndYearState & local government amountPercent of GDP.html",
                 "../outputs/State & local government amount/MapPlotsByVariableAndYearState & local government amountPercent of General Revenue.html",
                "../outputs/State & local government amount/MapPlotsByVariableAndYearState & local government amountValue Per Capita.html",
                 "../outputs/State & local government amount/LinePlotsStateFinancesAsPercentRevenuePercentGDPAndPerCapitaFigsState & local government amount.html",
                 "../outputs/State & local government amount/ScatterPlotsIncomePropertyAssessmentSalesFuelTaxesPctTotalRevenueState & local government amountPercent of General Revenue.html",
                "../outputs/State & local government amount/ScatterPlotsIncomePropertyAssessmentSalesFuelTaxesPctTotalRevenueState & local government amountPercent of GDP.html",
                 ]
for file in files_to_copy:
    copyfile(file, file.split("/")[-1])

os.system("jupyter nbconvert --to html ImportTemplates.ipynb")