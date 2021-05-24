import json
import torch
from utils import preprocess_data


device = torch.device('cpu')


def evaluation(question_dicts,file_write):
    print('Loading model...')
    with open('../data/vocab.txt', 'r') as f:
        vocab = json.load(f)
    model = torch.load('../models/trained_model.pt')


    preprocessed_questions, all_question_numbers, all_left_word_indices = preprocess_data(vocab, question_dicts)
    correct = 0.0
    eval_dict = []
    for i in range(len(preprocessed_questions)):
        eval_dict.append(question_dicts[i])
        preprocessed_question = preprocessed_questions[i]
        question_numbers = all_question_numbers[i]
        left_word_indices = all_left_word_indices[i]
        answer = torch.tensor(question_dicts[i]['answer']).to(device)
        is_scalar = question_dicts[i]['answer_type']
        table = torch.tensor(question_dicts[i]['table']).t().to(device)
        # print(question_dicts[i])
        # print(preprocessed_question, question_numbers, left_word_indices)

        guess = model(preprocessed_question, question_numbers, left_word_indices, table, mode='eval')
        # print("guess is ", guess)
        # print("answer is ", answer)
        # print('The question is:', question_dicts[i]['question'])
        eval_dict[i]['answer'] = int(guess[0])
        if (is_scalar and not guess[1]) or (not is_scalar and guess[1]) or (not torch.eq(answer.float(), guess[0])):
            print('False, the correct answer is', answer, 'but the guess is', guess)
        else:
            correct+=1
            print('Correct!')
        print('=' * 20)

    with open(file_write, 'w') as fp:
        json.dump(eval_dict, fp)
    print("The accuracy is ",correct/len(preprocessed_questions))


if __name__ == '__main__':
    with open('../data/split_testing_data_10.txt', 'r') as f_10:
        question_dicts_10 = json.load(f_10)
    file_write_10 = '../data/testing_eval_10.txt'
    evaluation(question_dicts_10,file_write_10)
    # with open('../data/split_testing_data_20.txt', 'r') as f_20:
    #     question_dicts_20 = json.load(f_20)
    # evaluation(question_dicts_20)

