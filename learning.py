import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence
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
                intermediate_steps = target_str.split(":")
                input_list = [int(char) for char in input_str]
                target_list = [[int(char) for char in intermediate_step] for intermediate_step in intermediate_steps]
                input_tensor = torch.tensor(input_list)
                target_tensor = torch.tensor(target_list)
                target_tensor = target_tensor.float()
                example = (input_tensor, target_tensor)
                self.examples.append(example)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index]

def collate_fn(batch):
    inputs, targets = zip(*batch)
    inputs = pad_sequence(inputs, batch_first=True, padding_value=-1)
    targets = pad_sequence(targets, batch_first=True, padding_value=-1)
    return inputs, targets

class SequenceInterpreter(nn.Module):

    def __init__(self, rnn_type, rnn_state_length, nb_rnn_layers, bidirectional, nb_cells, include_intermediate_steps):
        super().__init__()
        self.rnn_type = rnn_type
        self.nb_cells = nb_cells
        self.include_intermediate_steps = include_intermediate_steps
        if rnn_type == "GRU":
            rnn_class = nn.GRU
        elif rnn_type == "LSTM":
            rnn_class = nn.LSTM
        self.rnn = rnn_class(10, rnn_state_length, nb_rnn_layers, batch_first=True, bidirectional=bidirectional)
        in_features = rnn_state_length
        if bidirectional:
            in_features *= 2
        self.linear = nn.Linear(in_features, nb_cells)

    def forward(self, x):
        lengths = (x != -1).sum(dim=1).cpu()
        x = nn.functional.one_hot(x.clamp(0), 10)
        x = x.float()
        x = pack_padded_sequence(x, lengths, True, False)
        rnn_outputs, state = self.rnn(x)
        if self.include_intermediate_steps:
            padded_rnn_outputs, lengths = pad_packed_sequence(rnn_outputs, True, -1)
            x = padded_rnn_outputs[:, self.nb_cells - 1:, :]
        else:
            if self.rnn_type == "GRU":
                x = state[-1]
            elif self.rnn_type == "LSTM":
                x = state[0][-1]
            x = x.unsqueeze(1)
        x = self.linear(x)
        return x

def logits_to_bits(outputs):
    return (outputs > 0).float()

def get_nb_correct_cells_in_batch(targets, outputs):
    outputs = logits_to_bits(outputs)
    return (targets == outputs).sum().item()

def get_nb_cells_in_dataset(dataloader):
    nb_cells_in_dataset = 0
    for inputs, targets in dataloader:
        nb_cells_in_batch = (targets != -1).sum().item()
        nb_cells_in_dataset += nb_cells_in_batch
    return nb_cells_in_dataset

def get_nb_correct_sequences_in_batch(targets, outputs):
    outputs = logits_to_bits(outputs)
    outputs[targets == -1] = -1
    return (targets == outputs).all(dim=(1, 2)).sum().item()

def get_nb_sequences_in_dataset(dataloader):
    return len(dataloader) * dataloader.batch_size

def train_the_model(dataloader, model, loss_function, optimizer, device, max_gradient_norm):
    print("training")
    model.train()
    total_loss = 0
    nb_correct_cells = 0
    nb_correct_sequences = 0
    max_grad_norm = 0
    total_grad_norm = 0
    for i, (inputs, targets) in enumerate(dataloader):
        inputs = inputs.to(device)
        targets = targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        mask = targets != -1
        loss = loss_function(outputs[mask], targets[mask])
        loss.backward()
        if max_gradient_norm is None:
            max_gradient_norm = float("inf")
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_gradient_norm)
        if grad_norm > max_grad_norm:
            max_grad_norm = grad_norm
        total_grad_norm += grad_norm
        optimizer.step()
        total_loss += loss.item()
        nb_correct_cells += get_nb_correct_cells_in_batch(targets, outputs)
        nb_correct_sequences += get_nb_correct_sequences_in_batch(targets, outputs)
        if (i + 1) % 10 == 0:
            print(".", end="", flush=True)
    average_loss = total_loss / len(dataloader)
    cell_wise_accuracy = nb_correct_cells / get_nb_cells_in_dataset(dataloader)
    sequence_wise_accuracy = nb_correct_sequences / get_nb_sequences_in_dataset(dataloader)
    average_grad_norm = total_grad_norm / len(dataloader)
    print()
    return average_loss, cell_wise_accuracy, sequence_wise_accuracy, max_grad_norm, average_grad_norm

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
            mask = targets != -1
            loss = loss_function(outputs[mask], targets[mask])
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
