import ablang
import pandas as pd
import torch
import torch.nn.functional as F
import time
from tqdm import tqdm


def data_from_csv():
    flag = input('Аминокислотные последовательности лежат в файле v_seq_alignment_aa.csv? \t y/n \t')
    if flag == 'n':
        file_name = input('Введите имя файла: ')
        if not file_name.endswith('.csv'):
            file_name = file_name + '.csv'
    else:
        file_name = 'v_seq_alignment_aa.csv'

    global data, heavy_ablang
    data = pd.read_csv(file_name)
    heavy_ablang = ablang.pretrained('heavy')


def embeddings_for_heavy():
    """
    Столбцы, которые должны быть в csv-файле:
    ['v_sequence_alignment_aa_heavy', 'v_call_heavy', 'v_call_light']
    """
    try:
        all_seq = data['v_sequence_alignment_aa_heavy'].tolist()
        count = input(f'Сколько последовательностей необходимо закодировать '
                      f'(введите число < {data.shape[0]} или "all"): \t')
        count = data.shape[0] if count == 'all' else int(count)
        seqs = all_seq[:count]

        embedding = [[] for _ in range(data.shape[0])]
        for i, _ in enumerate(tqdm(seqs)):
            embedding[i] = heavy_ablang(seqs[i], mode='seqcoding')

        data['embedding_heavy'] = embedding

        try:
            data.to_csv('embeddings.cvs',
                        columns=['embedding_heavy', 'v_call_heavy', 'v_call_light'], index=False)

        except Exception as except_j:
            print('В файле должны быть столбцы ["v_call_heavy", "v_call_light"].')
            print(except_j)
    except Exception as except_i:
        print('А/к последовательности должны быть в столбце "v_sequence_alignment_aa_heavy".')
        print(except_i)


def cos_similarity():
    """
    Функция для проверки косинусного подобия.
    """
    columns = data.columns
    type_heavy = data['v_call_heavy'].unique()
    count_types = len(type_heavy)
    samples = [[] for _ in range(count_types)]
    idx = {type_heavy[i]: i for i in range(count_types)}
    count = {type_heavy[i]: 0 for i in range(count_types)}
    for i in range(data.shape[0]):
        type_i, seq_i = data[columns[2]][i], data[columns[0]][i]
        if count[type_i] > 2:
            continue
        samples[idx[type_i]].append(seq_i)
        count[type_i] += 1

    embeddings = [[] for _ in range(count_types)]
    for i, sample in enumerate(samples):
        embeddings[i] = heavy_ablang(sample, mode='seqcoding')

    for i in range(count_types):
        for j in range(i, count_types):
            a = torch.FloatTensor(embeddings[i][0])
            b = torch.FloatTensor(embeddings[j][1])
            result = F.cosine_similarity(a, b, dim=0)
            if i == j:
                print(f'\nКосинусное подобие для векторов одного типа {type_heavy[i]} равно {result:.2f}\n')
            else:
                print(f'Косинусное подобие для пары {type_heavy[i]} - {type_heavy[j]} равно {result:.2f}')
                a = torch.FloatTensor(embeddings[i][1])
                b = torch.FloatTensor(embeddings[j][0])
                result = F.cosine_similarity(a, b, dim=0)
                print(f'Для обратной пары: \t\t\t{type_heavy[j]} - {type_heavy[i]} равно {result:.2f}')


if __name__ == '__main__':
    print('\n' + '-' * 100)
    print('Загрузка данных.')
    data_from_csv()
    print('\n' + '-' * 100)
    print('Построение эмбеддингов.')
    embeddings_for_heavy()
    print('\n' + '-' * 100)
    print('Эмбеддинги для тяжёлых цепей и типы тяжёлых/лёгких цепей сохранены в файле "embeddings.cvs".')
    print('Названия столбцов файла: ["embedding_heavy", "v_call_heavy", "v_call_light"]')
