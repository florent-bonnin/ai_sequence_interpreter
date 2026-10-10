import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_sequence
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

def collate_fn(batch):
    inputs, targets = zip(*batch)
    inputs = pad_sequence(inputs, batch_first=True, padding_value=-1)
    targets = torch.stack(targets)
    return inputs, targets

class SequenceInterpreter(nn.Module):

    def __init__(self, rnn_type, rnn_state_length, nb_rnn_layers, bidirectional, nb_cells):
        super().__init__()
        self.rnn_type = rnn_type
        if rnn_type == "GRU":
            rnn_class = nn.GRU
        elif rnn_type == "LSTM":
            rnn_class = nn.LSTM
        self.rnn = rnn_class(1, rnn_state_length, nb_rnn_layers, batch_first=True, bidirectional=bidirectional)
        self.linear = nn.Linear(rnn_state_length, nb_cells)

    def forward(self, x):
        lengths = (x.squeeze(2) != -1).sum(dim=1).cpu()
        x = pack_padded_sequence(x, lengths, True, False)
        rnn_outputs, state = self.rnn(x)
        if self.rnn_type == "GRU":
            x = state[-1]
        elif self.rnn_type == "LSTM":
            x = state[0][-1]
        x = self.linear(x)
        return x

def logits_to_bits(outputs):
    return (outputs > 0).float()

def get_nb_correct_cells_in_batch(targets, outputs):
    outputs = logits_to_bits(outputs)
    return (targets == outputs).sum().item()

def get_nb_cells_in_dataset(dataloader):
    nb_examples = len(dataloader) * dataloader.batch_size
    dataset = dataloader.dataset
    input_tensor, target_tensor = dataset[0]
    nb_cells_per_example = target_tensor.numel()
    return nb_examples * nb_cells_per_example

def get_nb_correct_sequences_in_batch(targets, outputs):
    outputs = logits_to_bits(outputs)
    return (targets == outputs).all(dim=1).sum().item()

def get_nb_sequences_in_dataset(dataloader):
    return len(dataloader) * dataloader.batch_size

def train_the_model(dataloader, model, loss_function, optimizer, device):
    print("training")
    model.train()
    total_loss = 0
    nb_correct_cells = 0
    nb_correct_sequences = 0
    for i, (inputs, targets) in enumerate(dataloader):
        inputs = inputs.to(device)
        targets = targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_function(outputs, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        nb_correct_cells += get_nb_correct_cells_in_batch(targets, outputs)
        nb_correct_sequences += get_nb_correct_sequences_in_batch(targets, outputs)
        if (i + 1) % 10 == 0:
            print(".", end="", flush=True)
    average_loss = total_loss / len(dataloader)
    cell_wise_accuracy = nb_correct_cells / get_nb_cells_in_dataset(dataloader)
    sequence_wise_accuracy = nb_correct_sequences / get_nb_sequences_in_dataset(dataloader)
    print()
    return average_loss, cell_wise_accuracy, sequence_wise_accuracy

def evaluate_the_model(dataloader, model, loss_function, device):
    print("evaluating")
    model.eval()
    total_loss = 0
    nb_correct_cells = 0
    nb_correct_sequences = 0
    with torch.no_grad():
        for i, (inputs, targets) in enumerate(dataloader):
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            loss = loss_function(outputs, targets)
            total_loss += loss.item()
            nb_correct_cells += get_nb_correct_cells_in_batch(targets, outputs)
            nb_correct_sequences += get_nb_correct_sequences_in_batch(targets, outputs)
            if (i + 1) % 10 == 0:
                print(".", end="", flush=True)
    average_loss = total_loss / len(dataloader)
    cell_wise_accuracy = nb_correct_cells / get_nb_cells_in_dataset(dataloader)
    sequence_wise_accuracy = nb_correct_sequences / get_nb_sequences_in_dataset(dataloader)
    print()
    return average_loss, cell_wise_accuracy, sequence_wise_accuracy
