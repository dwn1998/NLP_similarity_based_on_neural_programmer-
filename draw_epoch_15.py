import json

import tqdm
import torch

from neural_programmer import NeuralProgrammer
from operations import OPERATIONS
from utils import build_vocab, preprocess_data
import matplotlib.pyplot as plt
import numpy as np

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')


def scalar_loss(guess, answer, huber):
    huber = torch.tensor(huber).to(device)

    a = abs(guess - answer)
    if a <= huber:
        return 0.5 * pow(a, 2)
    else:
        return huber * a - 0.5 * pow(huber, 2)


def lookup_loss(guess: torch.Tensor, answer: torch.Tensor):
    assert guess.size() == answer.size()

    return (-1 / (torch.tensor(answer.size(0), dtype=torch.float)) * (
        torch.tensor(answer.size(1), dtype=torch.float))) * sum(
        [
            answer[i][j] * torch.log(guess[i][j] + torch.tensor(0.00000001)) +
            (torch.tensor(1.) - answer[i][j]) * torch.log(torch.tensor(1.) - guess[i][j])
            for j in range(answer.size(1)) for i in range(answer.size(0))
        ]
    )


def loss_fn(scalar_guess, lookup_guess, answer, is_scalar):
    return scalar_loss(scalar_guess, answer, 10.) if is_scalar else lookup_loss(lookup_guess, answer)


def train(question_dicts,flag):
    # TODO implement a mini-batch technique for training
    print('Pre-processing the questions...')
    vocab = build_vocab(question_dicts)
    with open('../data/vocab.txt', 'w') as f:
        json.dump(vocab, f)

    preprocessed_questions, all_question_numbers, all_left_word_indices = \
        preprocess_data(vocab, question_dicts)

    model = NeuralProgrammer(256, len(vocab), len(OPERATIONS), 1, 4)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    print('Starting to train...')
    Y_train = []
    for epoch in range(15):
        total_loss = 0.0
        for i in tqdm.tqdm(range(len(preprocessed_questions))):
            preprocessed_question = preprocessed_questions[i]
            question_numbers = all_question_numbers[i]
            left_word_indices = all_left_word_indices[i]
            answer = torch.tensor(question_dicts[i]['answer']).to(device)
            # is_scalar = torch.tensor(question_dicts[i]['answer_type'])
            table = torch.tensor(question_dicts[i]['table']).t().to(device)

            scalar_guess, lookup_guess = model(preprocessed_question, question_numbers, left_word_indices, table,
                                               mode='train')
            loss = scalar_loss(scalar_guess, answer, 10.)
            total_loss += loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        avg_loss = total_loss / len(preprocessed_question)
        Y_train.append(avg_loss.detach().numpy())
        print(('avg loss at epoch %d: ' % epoch), total_loss / len(preprocessed_question))

        # if epoch % 5 == 0 and epoch != 0 and flag==1:
        #     print('Saving model at epoch %d' % epoch)
        #     torch.save(model, '../models/trained_model_epoch%d.pt' % epoch)

    if flag == 1:
        torch.save(model, '../models/trained_model_15.pt')
    return Y_train


if __name__ == '__main__':
    print('Loading dataset...')
    file_name_train = '../data/split_training_data_10.txt'
    file_name_test_10 = '../data/split_testing_data_10.txt'
    file_name_test_20 = '../data/split_testing_data_20.txt'
    
    with open(file_name_train, 'r') as f_1:
        question_dicts_train = json.load(f_1)
    Y_training = train(question_dicts_train,1)
    with open(file_name_test_10, 'r') as f_2:
        question_dicts_test_10 = json.load(f_2)
    Y_testing_10 = train(question_dicts_test_10,0)

    with open(file_name_test_20, 'r') as f_3:
        question_dicts_test_20 = json.load(f_3)
    Y_testing_20 = train(question_dicts_test_20,0)
    print(Y_training)
    print(Y_testing_10)
    print(Y_testing_20)

    x = np.arange(0,15)
    plt.title("learning curve during training in epoch 15")
    plt.xlabel("training iteration") 
    plt.ylabel("loss") 
    plt.plot(x,Y_training,label="training") 
    plt.plot(x,Y_testing_10,label="testing_10") 
    plt.plot(x,Y_testing_20,label="testing_20") 
    plt.legend(ncol=3)
    plt.show()
