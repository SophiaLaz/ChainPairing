import pandas as pd

df = pd.read_csv("all_pared.csv")
table = df.value_counts(['v_call_heavy', 'v_call_light'])
table = table.to_frame().reset_index().rename(columns={0: 'count'})
table = table.sort_values(by=['v_call_heavy', 'v_call_light'])
table = table.pivot_table(index='v_call_light', columns='v_call_heavy', values='count', fill_value=0)
table.to_csv('conjugacy_table.csv')
