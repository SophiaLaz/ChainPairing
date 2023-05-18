import traceback
import ablang
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.utils import gen_batches
from tqdm import tqdm
import json


def data_from_csv():
    flag = input('Аминокислотные последовательности лежат в файле v_seq_alignment_aa.csv? \t y/n \t')
    if flag == 'n':
        file_name = input('Введите имя файла: ')
        if not file_name.endswith('.csv'):
            file_name = file_name + '.csv'
    else:
        file_name = 'v_seq_alignment_aa.csv'

    data = pd.read_csv(file_name)
    print('\n' + '-' * 100)
    print('Фильтр датасета от повторений.\n')
    return del_duplicate(data)


def del_duplicate(data):
    print(f'Количество строк в изначальном датасете: \t{data.shape[0]}.')
    alphabet, index_for_del = set(), []
    for i, seq in enumerate(data['v_sequence_alignment_aa_heavy']):
        if seq in alphabet:
            index_for_del.append(i)
        else:
            alphabet.add(seq)
    data.drop(index=index_for_del, inplace=True)
    print(f'Удалено строк: \t\t\t\t\t\t\t\t{len(index_for_del)}.')
    print(f'\nКоличество строк после удаления дубликатов: {data.shape[0]}.')
    return data


def embeddings_for_heavy(data):
    """
    Столбцы, которые должны быть в csv-файле:
    ['v_sequence_alignment_aa_heavy', 'v_call_heavy', 'v_call_light']
    """
    heavy_ablang = ablang.pretrained('heavy')
    try:
        all_seq = data['v_sequence_alignment_aa_heavy'].tolist()
        count = input(f'Сколько последовательностей необходимо закодировать '
                      f'(введите число < {data.shape[0]} или "all"): \t')
        count = data.shape[0] if count == 'all' else int(count)
        seqs = all_seq[:count]

        embedding = [[] for _ in range(data.shape[0])]
        batch_size = 50
        for i, batch in enumerate(tqdm(list(gen_batches(count, batch_size)))):
            embedding[batch] = heavy_ablang(seqs[batch], mode='seqcoding').tolist()

        data['embedding_heavy'] = embedding
        seq_heavy = data['v_sequence_alignment_aa_heavy'].tolist()
        seq_light = data['v_sequence_alignment_aa_light'].tolist()
        seq_embedding = {seq_heavy[i]: embedding[i] for i in range(data.shape[0])}
        seq_heavy_light = {seq_heavy[i]: seq_light[i] for i in range(data.shape[0])}

        with open('embeddings.json', 'w') as file:
            json.dump(seq_embedding, file)
        with open('paired.json', 'w') as file:
            json.dump(seq_heavy_light, file)
        print('Эмбеддинги успешно сохранены в файле "embeddings.json".')
        print('Комплиментарные последовательности "v_sequence_alignment_aa_heavy" '
              '\n\tи "v_sequence_alignment_aa_light" сохранены в файле "paired.json".')

    except Exception as except_i:
        print('А/к последовательности должны быть в столбце "v_sequence_alignment_aa_heavy".')
        traceback.print_exception(except_i)


def cos_similarity(data):
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
    heavy_ablang = ablang.pretrained('heavy')
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
    data_set = data_from_csv()
    print('\n' + '-' * 100)
    print('Построение эмбеддингов.')
    embeddings_for_heavy(data_set)
    print('\n' + '-' * 100)
    print('Эмбеддинги для тяжёлых цепей и типы тяжёлых/лёгких цепей сохранены в файле "embeddings.csv".')
    print('Названия столбцов файла: ["embedding_heavy", "v_call_heavy", "v_call_light"]')
