from keplar.data.data import Data
from keplar.preoperator.ml.sklearndbscan import SklearnDBscan



data = Data("txt", "datasets/2.txt", ["x0", "x1", "x2", "x3", "x4", "y"])
data.read_file()
data.set_xy("y")

for i in [1e-5, 0.2, 1, 4, 10, 100]:
    dbscan = SklearnDBscan(eps=i)
    x = dbscan.do(data)
    if x:
        break
db_sum = x
for db_i in db_sum:

