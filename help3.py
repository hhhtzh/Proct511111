# str="c0011"
# str1=str[1:]
# print(int(str1))
from pmlb import fetch_data
import pyoperon as Operon

D=fetch_data('1027_ESL', return_X_y=False, local_cache_dir='./datasets').to_numpy()
# # initialize a dataset from a numpy array
ds = Operon.Dataset(D)
vr=ds.Variables
var_dict={}
for var in vr:
    var_dict[int(var.Hash)]=str(var.Name)
print(var_dict[143321629840518241])
