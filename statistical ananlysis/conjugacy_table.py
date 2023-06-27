import pandas as pd

file_names = ['v_call.csv', 'j_call.csv']
columns = [['v_call_heavy', 'v_call_light'], ['j_call_heavy', 'j_call_light']]

for file_name, column in zip(file_names, columns):
    df = pd.read_csv(file_name)

    table = df.value_counts(column)
    table = table.to_frame().reset_index().rename(columns={0: 'count'})
    table = table.sort_values(by=column)
    table = table.pivot_table(index=column[1], columns=column[0], values='count', fill_value=0)
    table.to_csv('conjugacy_table/' + file_name)

columns = [['v_call_heavy', 'j_call_light'], ['j_call_heavy', 'v_call_light']]
file_name = 'v_and_j_call.csv'
table_names = ['vh_and_jl.csv', 'jh_and_vl.csv']
for column, table_name in zip(columns, table_names):
    df = pd.read_csv(file_name)
    table = df.value_counts(column)
    table = table.to_frame().reset_index().rename(columns={0: 'count'})
    table = table.sort_values(by=column)
    table = table.pivot_table(index=column[1], columns=column[0], values='count', fill_value=0)
    table.to_csv('conjugacy_table/' + table_name)
