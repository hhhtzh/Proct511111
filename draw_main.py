import pandas as pd

from keplar.draw.BoxImg import draw_box_diagram, addLog

Vladislavleva = pd.read_csv('result/Vladislavleva_result.csv')
# draw_box_diagram(addLog(Vladislavleva['GpBingo']), addLog(Vladislavleva['Bingo']), addLog(Vladislavleva['OperonBingo']),
#                  addLog(Vladislavleva['Operon']), addLog(Vladislavleva['MTaylorGp']), addLog(Vladislavleva['MTaylorBingo']),
#                  addLog(Vladislavleva['MTaylorOperon']), addLog(Vladislavleva['TaylorBingo']), addLog(Vladislavleva['BingoCPP']), addLog(Vladislavleva['UDSR']),
#                  "vlsdislavleva", isAll=True)
draw_box_diagram(Vladislavleva['GpBingo'], Vladislavleva['Bingo'], Vladislavleva['OperonBingo'],
                 Vladislavleva['Operon'], Vladislavleva['MTaylorGp'], Vladislavleva['MTaylorBingo'],
                 Vladislavleva['MTaylorOperon'], Vladislavleva['TaylorBingo'], Vladislavleva['BingoCPP'], Vladislavleva['MTaylorKMeans'],
                 "vlsdislavleva", isAll=True)

# draw_box_diagram(addLog(GECCO['TaylorGP']), addLog(GECCO['GPLearn']), addLog(GECCO['FFX']),
#                  addLog(GECCO['LinearRegression']), addLog(GECCO['KernelRidge']), addLog(GECCO['RF']),
#                  addLog(GECCO['SVM']), addLog(GECCO['XGBoost']), addLog(GECCO['GSGP']), addLog(GECCO['BSR']),
#                  "GECCO", isAll=False)

# draw_box_diagram_inside(addLog(GECCO['TaylorGP']), addLog(GECCO['GPLearn']), addLog(GECCO['FFX']),
#                         addLog(GECCO['LinearRegression']), addLog(GECCO['KernelRidge']), addLog(GECCO['RF']),
#                         addLog(GECCO['SVM']), addLog(GECCO['XGBoost']), addLog(GECCO['GSGP']), addLog(GECCO['BSR']),
#                         "GECCO")