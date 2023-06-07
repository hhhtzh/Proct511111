# str="c0011"
# str1=str[1:]
# print(int(str1))
import numpy as np
from pmlb import fetch_data
import pyoperon as Operon

from keplar.operator.creator import OperonCreator

x, y = fetch_data('1027_ESL', return_X_y=True, local_cache_dir='./datasets')
# # initialize a dataset from a numpy array
x = np.array(x)
y = np.array(y)
# vr=ds.Variables
# var_dict={}
# for var in vr:
#     var_dict[int(var.Hash)]=str(var.Name)
# print(var_dict[143321629840518241])
cr = OperonCreator("balanced", x, y, 128,"self")
pop = cr.do()
