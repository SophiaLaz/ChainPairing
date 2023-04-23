import pandas as pd
import os

parth = 'data_set/paired/'
filenames = os.listdir(parth)
df = pd.DataFrame()
for file_i in filenames:
    try:
        df_i = pd.read_csv(parth+file_i, skiprows=1, usecols=['v_call_heavy', 'v_call_light'])
        df_i = df_i.dropna()
        mask1 = df_i['v_call_heavy'].apply(lambda x: x.split(sep='-')[0][:-1] == 'IGHV')
        mask2 = df_i['v_call_light'].apply(lambda x: x.split(sep='-')[0][:3] in ['IGK', 'IGL'])
        df_i = df_i[mask1 & mask2]
        df_i['v_call_heavy'] = df_i['v_call_heavy'].str.split('-').str[0]
        df_i['v_call_light'] = df_i['v_call_light'].str.split('-').str[0]
        df = pd.concat([df, df_i], ignore_index=True)
    except Exception as except_i:
        print(f'File {file_i} is not correct: {except_i}')

df.to_csv('all_pared.csv', index=False)
