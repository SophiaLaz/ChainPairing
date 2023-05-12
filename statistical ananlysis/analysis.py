import pandas as pd
import scipy.stats as st


file_names, titles = ['v_call.csv', 'j_call.csv'], ['v-гена', 'j-гена']

for file_name, title in zip(file_names, titles):
    table_df = pd.read_csv('conjugacy_table/' + file_name)
    table_df = table_df.drop(table_df.columns[0], axis=1)
    table = table_df.values.tolist()

    stat, p, dof, expected = st.chi2_contingency(table)
    prob = 0.95
    critical = st.chi2.ppf(prob, dof)

    # print(f'Статистичсекий анализ критерием Пирсона для {title}:')
    # if abs(stat) >= critical:
    #     print('Зависимые (отвергаем гипотезу H0)')
    # else:
    #     print('Независимые (не отвергаем гипотезу H0)')

    print(f'\nСтатистичсекий анализ критерием Крамера для {title}:')
    r, c = table_df.shape
    n = table_df.values.sum()
    cramer = (stat / (n * min(c-1, r-1))) ** 0.5

    if cramer > 0.5:
        print(f'Коэффициент ближе к единице (V = {cramer:.4f}), значит присутствует корреляция.')
    else:
        print(f'Коэффициент ближе к нулю (V = {cramer:.4f}), значит корреляции нет.')
