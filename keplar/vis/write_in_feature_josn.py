import pandas as pd


ft = pd.read_feather("pmlb_results.feather")
new_row = pd.read_json("feaure.josn")
ft = pd.concat([ft, new_row], ignore_index=True)
ft.to_feather("pmlb_results.feather")


