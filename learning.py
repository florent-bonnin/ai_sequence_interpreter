import torch
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
                input_tensor = torch.tensor(input_list)
                target_tensor = torch.tensor(target_list)
                example = (input_tensor, target_tensor)
                self.examples.append(example)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index]
