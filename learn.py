from learning import SequenceInterpreter
from learning import SequenceInterpreterDataset
import torch
from torch.utils.data import DataLoader

BATCH_SIZE = 1
DATASET_PATH = "datasets/dev"
NB_RNN_LAYERS = 1
RNN_STATE_LENGTH = 100
RNN_TYPE = "GRU"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"{device}\n")

train_dataset = SequenceInterpreterDataset(f"{DATASET_PATH}/train.csv")
val_dataset = SequenceInterpreterDataset(f"{DATASET_PATH}/val.csv")
test_dataset = SequenceInterpreterDataset(f"{DATASET_PATH}/test.csv")

train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

input_tensor, target_tensor = train_dataset[0]
nb_cells = target_tensor.shape[0]
sequence_interpreter = SequenceInterpreter(RNN_TYPE, RNN_STATE_LENGTH, NB_RNN_LAYERS, nb_cells)
print(f"{sequence_interpreter}")
sequence_interpreter.to(device)

input_batch, target_batch = next(iter(train_dataloader))
print(input_batch)
input_batch = input_batch.to(device)
output = sequence_interpreter(input_batch)
print(output)
