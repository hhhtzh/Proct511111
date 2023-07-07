from keplar.data.data import Data
from keplar.preoperator.feature_engineering import TaylorFeature

data = Data("pmlb", "1027_ESL", ["x1", "x2", "x3", "x4", 'y'])
data.read_file()
ty=TaylorFeature(data,"test1")
taylor_metric=ty.do()