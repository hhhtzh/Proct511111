import pandas as pd

ft = pd.read_feather("pmlb_results.feather")
# print(ft.iloc[-1])
# print(ft.loc[ft["dataset"] == "feynman-ii.38.3"]["r2_test"])
# print(ft.loc[ft["algorithm"] == "mtaylor"]["r2_test"])
print(ft.loc[ft["algorithm"] == "Operon"])