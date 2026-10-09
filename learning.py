import torch
from torch import nn
from torch.utils.data import Dataset

class SequenceInterpreterDataset(Dataset):

    def __init__(self, path):
        self.examples = []
        with open(path, "r", encoding="utf-8") as file:
            for line in file:
                line = line[:-1]
                fields = line.split(",")
                input_str = fields[0]
                target_str = fields[1]
                input_list = [int(char) for char in input_str]
                target_list = [int(char) for char in target_str]
                input_tensor = torch.tensor(input_list).float()
                target_tensor = torch.tensor(target_list).float()
                input_tensor = input_tensor.unsqueeze(1)
                example = (input_tensor, target_tensor)
                self.examples.append(example)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index]

class SequenceInterpreter(nn.Module):

    def __init__(self, rnn_type, rnn_state_length, nb_rnn_layers, nb_cells):
        super().__init__()
        if rnn_type == "GRU":
            rnn_class = nn.GRU
        elif rnn_type == "LSTM":
            rnn_class = nn.LSTM
        self.rnn = rnn_class(1, rnn_state_length, nb_rnn_layers, batch_first=True)
        self.linear = nn.Linear(rnn_state_length, nb_cells)

    def forward(self, x):
        rnn_outputs, state = self.rnn(x)
        x = rnn_outputs[:, -1, :]
        x = self.linear(x)
        return x
