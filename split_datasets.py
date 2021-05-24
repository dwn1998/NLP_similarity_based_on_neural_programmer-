import json
import sys
import os

if __name__ == '__main__':
    file_name = 'training_set_1000.txt'
    with open(file_name, 'r') as f:
        question_dicts = json.load(f)
    train_dicts = []
    train_10_dict = []
    test_10_dict = []
    test_20_dict = []
    for question_dict in question_dicts:
        if len(question_dict['table'][0])<11:
            train_dicts.append(question_dict)
        else:
            test_20_dict.append(question_dict)
    l = len(train_dicts)
    for i in range(int(l*0.8)):
        train_10_dict.append(train_dicts[i])
    for i in range(int(l*0.8),l):
        test_10_dict.append(train_dicts[i])


    with open("split_training_data_10.txt", 'w') as fp:
        json.dump(train_10_dict, fp)

    with open("split_testing_data_10.txt", 'w') as fp:
        json.dump(test_10_dict, fp)

    with open("split_testing_data_20.txt", 'w') as fp:
        json.dump(test_20_dict, fp)