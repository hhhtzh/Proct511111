from keplar.data.data import Data
from keplar.operator.feature_engineering import TaylorFeature
from keplar.operator.taylor_judge import TaylorJudge

data = Data("pmlb", "1027_ESL", ["x1", "x2", "x3", "x4", 'y'])
data.read_file()
ty=TaylorJudge(data,"taylorgp")
ty.do()