import pandas as pd

from keplar.utils.BoxImg import draw_box_diagram, addLog, draw_time_box_diagram
from keplar.draw.BoxImg_vla_5 import draw_box_diagram_vla_5, draw_time_box_diagram_vla_5

Vladislavleva = pd.read_csv('result/vla_5.csv')
# draw_box_diagram(addLog(Vladislavleva['GpBingo']), addLog(Vladislavleva['Bingo']), addLog(Vladislavleva['OperonBingo']),
#                  addLog(Vladislavleva['Operon']), addLog(Vladislavleva['MTaylorGp']), addLog(Vladislavleva['MTaylorBingo']),
#                  addLog(Vladislavleva['MTaylorOperon']), addLog(Vladislavleva['TaylorBingo']), addLog(Vladislavleva['BingoCPP']), addLog(Vladislavleva['UDSR']),
#                  "vlsdislavleva", isAll=True)
draw_box_diagram_vla_5(Vladislavleva['Bingo'], Vladislavleva['Gplearn'], Vladislavleva['BingoCPP'],
                       Vladislavleva['KeplarBingoCPP'], Vladislavleva['KeplarBingo'], Vladislavleva['Operon'],
                       Vladislavleva['KeplarMBingo'], Vladislavleva['KeplarMOperon'], Vladislavleva['GpBingo'],
                       Vladislavleva['TaylorBingo'], Vladislavleva['uDSR'], Vladislavleva['mtaylorKMeans'],
                       Vladislavleva['OperonBingo'], Vladislavleva['MTaylor'],
                       "vla_5", isAll=True)
pmlb_time = pd.read_csv('result/vla_5_time.csv')
draw_time_box_diagram_vla_5(pmlb_time['Bingo'], pmlb_time['Gplearn'], pmlb_time['BingoCPP'],
                            pmlb_time['KeplarBingoCPP'], pmlb_time['KeplarBingo'], pmlb_time['Operon'],
                            pmlb_time['KeplarMBingo'], pmlb_time['KeplarMOperon'], pmlb_time['GpBingo'],
                            pmlb_time['TaylorBingo'], pmlb_time['uDSR'], pmlb_time['MTaylorKmeans'],
                            pmlb_time['OperonBingo'], pmlb_time['MTaylor'],
                            "vla_5", isAll=True)
