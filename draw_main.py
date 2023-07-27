import pandas as pd

from keplar.draw.BoxImg import draw_box_diagram, addLog, draw_time_box_diagram

Vladislavleva = pd.read_csv('result/pmlb_1027_result.csv')
# draw_box_diagram(addLog(Vladislavleva['GpBingo']), addLog(Vladislavleva['Bingo']), addLog(Vladislavleva['OperonBingo']),
#                  addLog(Vladislavleva['Operon']), addLog(Vladislavleva['MTaylorGp']), addLog(Vladislavleva['MTaylorBingo']),
#                  addLog(Vladislavleva['MTaylorOperon']), addLog(Vladislavleva['TaylorBingo']), addLog(Vladislavleva['BingoCPP']), addLog(Vladislavleva['UDSR']),
#                  "vlsdislavleva", isAll=True)
draw_box_diagram(Vladislavleva['GpBingo'], Vladislavleva['Bingo'], Vladislavleva['OperonBingo'],
                 Vladislavleva['Operon'], Vladislavleva['MTaylor'], Vladislavleva['KeplarBingo'],
                 Vladislavleva['KeplarBingoCPP'], Vladislavleva['TaylorBingo'], Vladislavleva['BingoCPP'],Vladislavleva['KeplarMBingo'],
                 "vlsdislavleva", isAll=True)

pmlb_time = pd.read_csv('result/pmlb_1027_time_result.csv')
draw_time_box_diagram(pmlb_time['GpBingo'], pmlb_time['Bingo'], pmlb_time['OperonBingo'],
                 pmlb_time['Operon'], pmlb_time['MTaylor'], pmlb_time['KeplarBingo'],
                 pmlb_time['KeplarBingoCPP'], pmlb_time['TaylorBingo'], pmlb_time['BingoCPP'],Vladislavleva['KeplarMBingo'],
                 "vlsdislavleva", isAll=True)

# draw_box_diagram(addLog(GECCO['TaylorGP']), addLog(GECCO['GPLearn']), addLog(GECCO['FFX']),
#                  addLog(GECCO['LinearRegression']), addLog(GECCO['KernelRidge']), addLog(GECCO['RF']),
#                  addLog(GECCO['SVM']), addLog(GECCO['XGBoost']), addLog(GECCO['GSGP']), addLog(GECCO['BSR']),
#                  "GECCO", isAll=False)

# draw_box_diagram_inside(addLog(GECCO['TaylorGP']), addLog(GECCO['GPLearn']), addLog(GECCO['FFX']),
#                         addLog(GECCO['LinearRegression']), addLog(GECCO['KernelRidge']), addLog(GECCO['RF']),
#                         addLog(GECCO['SVM']), addLog(GECCO['XGBoost']), addLog(GECCO['GSGP']), addLog(GECCO['BSR']),
#                         "GECCO")