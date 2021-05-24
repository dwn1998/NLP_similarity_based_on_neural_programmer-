import json
import torch
from utils import preprocess_data


device = torch.device('cpu')

def tf_similarity_original(original_dataset_dicts1, original_dataset_dicts2):
    s1 = original_dataset_dicts1['table'][0]
    for e in original_dataset_dicts1['question'].split(' '):
        s1.append(e)
    s2 = original_dataset_dicts2['table'][0]
    for e in original_dataset_dicts2['question'].split(' '):
        s2.append(e)
    number_same = 0.0
    if len(s1)>len(s2):
        for e in s1:
            if (e in s2):
                number_same+=1
        return number_same / len(s1)
    else:
        for e in s2:
            if (e in s1):
                number_same+=1
        return number_same / len(s2)

def tf_similarity_eval(eval_dicts1, eval_dicts2):
    s1 = eval_dicts1['table'][0]
    for e in eval_dicts1['question'].split(' '):
        s1.append(e)
    s2 = eval_dicts2['table'][0]
    for e in eval_dicts2['question'].split(' '):
        s2.append(e)
    number_same = 0.0
    print(s1,s2)
    l1 = len(s1)
    l2 = len(s2)
    if l1>l2:
        for e in s1:
            if (e in s2):
                number_same+=1
        if eval_dicts1['answer']==eval_dicts2['answer']:
            number_same+=l1
        return number_same / (l1*2)
    else:
        for e in s2:
            if (e in s1):
                number_same+=1
        if eval_dicts1['answer']==eval_dicts2['answer']:
            number_same+=l2
        return number_same / (l2*2)

def evaluation(eval_dicts,original_dataset_dicts,file_write):
    write_dicts = []
    for i in range(len(eval_dicts)):
        write_dict = dict()
        write_dict['q1_table'] = eval_dicts[i]['table'] 
        write_dict['q2_table'] = original_dataset_dicts[i]['table'] 
        write_dict['similarity_with_eval'] = tf_similarity_eval(eval_dicts[i],original_dataset_dicts[i])
        write_dict['similarity_of_original_dataset'] = tf_similarity_original(eval_dicts[i],original_dataset_dicts[i])
        write_dicts.append(write_dict)
    with open(file_write, 'w') as fp:
        json.dump(write_dicts, fp)

if __name__ == '__main__':
    with open('../data/testing_eval_10.txt', 'r') as f_10:
        eval_dicts = json.load(f_10)
    with open('../data/split_testing_data_10.txt', 'r') as f_10:
        original_dataset_dicts = json.load(f_10)
    file_write = '../data/similarity.txt'
    evaluation(eval_dicts,original_dataset_dicts,file_write)

    # with open('../data/split_testing_data_20.txt', 'r') as f_20:
    #     question_dicts_20 = json.load(f_20)
    # evaluation(question_dicts_20)

