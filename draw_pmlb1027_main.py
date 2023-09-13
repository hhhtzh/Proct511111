import pandas as pd

from keplar.draw.draw import BoxImgDraw
from keplar.utils.BoxImg import draw_box_diagram, addLog, draw_time_box_diagram

Vladislavleva = pd.read_csv('result/pmlb_1027_result.csv')
dbi = BoxImgDraw("csv", 'result/pmlb_1027_result.csv', ['KeplarBingo', 'Bingo', 'Gplearn',
                                                        'KeplarMBingo', 'Operon', 'KeplarBingo', 'KeplarBingoCPP',
                                                        'TaylorBingo', 'BingoCPP', 'KeplarMBingo', 'KeplarMOperon',
                                                        'MTaylor(With new Sparse)'], "pmlb_1027")
dbi.draw()
