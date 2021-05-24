import torch
import torch.nn as nn


# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')


class QuestionRNN(nn.Module):
    def __init__(self, hidden_dim: int, vocab_size: int):
        super(QuestionRNN, self).__init__()
        self.hidden_dim = hidden_dim

        self.word_embedding = nn.Embedding(vocab_size, hidden_dim)
        self.input2hidden = nn.Linear(2 * hidden_dim, hidden_dim)
        self.tanh = nn.Tanh()

        self.hidden_states = []

    def forward(self, input_question: [int]):
        hidden_state = torch.zeros(self.hidden_dim).to(device)
        self.hidden_states.append(hidden_state)
        for word_index in input_question:
            embedded_word = self.word_embedding(torch.tensor(word_index).to(device))
            hidden_state = self.tanh(self.input2hidden(torch.cat((hidden_state, embedded_word))))
            self.hidden_states.append(hidden_state)

        return hidden_state
