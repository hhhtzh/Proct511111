from keplar.population.function import function_map_dsr, operator_map,operator_map_dsr,operator_map_dsr2

print(operator_map_dsr2)

print(function_map_dsr)
for i in operator_map_dsr2:
    # print(operator_map_dsr2[i])
    if function_map_dsr.get(operator_map_dsr2[i]) != None:
        operator_map_dsr2[i] = function_map_dsr[operator_map_dsr2[i]]
#     for j in range(len(function_map_dsr)):
#         if operator_map_dsr2[i] == str(function_map_dsr[j].keys()):
#             # operator_map_dsr2[i] = operator_map[j]
#             print(operator_map_dsr2[i])

            # print(operator_map_dsr2[i])
    # if operator_map_dsr2[i]  function_map_dsr.keys():
        # operator_map_dsr2[i] = function_map_dsr[operator_map_dsr2[i]]