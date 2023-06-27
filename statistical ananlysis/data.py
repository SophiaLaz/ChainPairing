import pandas as pd
import os

parth = 'data_set/paired/'
filenames = os.listdir(parth)
df = pd.DataFrame()
columns = ['v_call_heavy', 'j_call_heavy', 'Isotype_heavy', 'v_sequence_alignment_aa_heavy',
           'v_call_light', 'j_call_light', 'Isotype_light', 'v_sequence_alignment_aa_light']
for file_i in filenames:
    try:
        df_i = pd.read_csv(parth+file_i, skiprows=1, usecols=columns)
        df_i = df_i.dropna()

        mask1 = df_i['v_call_heavy'].apply(lambda x: x.split(sep='-')[0][:-1] == 'IGHV')
        mask2 = df_i['v_call_light'].apply(lambda x: x.split(sep='-')[0][:3] in ['IGK', 'IGL'])
        mask3 = df_i['j_call_heavy'].apply(lambda x: x.split(sep='*')[0][:-1] == 'IGHJ')
        mask4 = df_i['j_call_light'].apply(lambda x: x.split(sep='*')[0][:-1] in ['IGKJ', 'IGLJ'])
        df_i = df_i[mask1 & mask2 & mask3 & mask4]
        df_i['v_call_heavy'] = df_i['v_call_heavy'].str.split('-').str[0]
        df_i['v_call_light'] = df_i['v_call_light'].str.split('-').str[0]
        df_i['j_call_heavy'] = df_i['j_call_heavy'].str.split('*').str[0]
        df_i['j_call_light'] = df_i['j_call_light'].str.split('*').str[0]
        df = pd.concat([df, df_i], ignore_index=True)

    except Exception as except_i:
        print(f'File {file_i} is not correct: {except_i}')

df.to_csv('v_call.csv', columns=['v_call_heavy', 'v_call_light'], index=False)
df.to_csv('j_call.csv', columns=['j_call_heavy', 'j_call_light'], index=False)
df.to_csv('isotype.csv', columns=['Isotype_heavy', 'Isotype_light'], index=False)
df.to_csv('v_seq_alignment_aa.csv',
          columns=['v_sequence_alignment_aa_heavy', 'v_sequence_alignment_aa_light',
                   'v_call_heavy', 'v_call_light', 'Isotype_heavy', 'Isotype_light'], index=False)
