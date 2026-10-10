from learning import evaluate_the_model
from learning import SequenceInterpreter
from learning import SequenceInterpreterDataset
from learning import train_the_model
from pathlib import Path
import torch
from torch import nn
from torch.utils.data import DataLoader

BATCH_SIZE = 64
DATASET_PATH = "datasets/dev"
NB_EPOCHS = 1000
NB_RNN_LAYERS = 1
RNN_STATE_LENGTH = 1000
RNN_TYPE = "GRU"
WEIGHT_DECAY = 0.1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"{device}\n")

# GETTING INFO
train_path = f"{DATASET_PATH}/train"
train_files = sorted([str(path) for path in Path(train_path).iterdir()])
nb_curriculum_steps = len(train_files)
print(f"nb_curriculum_steps = {nb_curriculum_steps}")

# MODEL CREATION
dataset = SequenceInterpreterDataset(train_files[0])
input_tensor, target_tensor = dataset[0]
nb_cells = target_tensor.shape[0]
sequence_interpreter = SequenceInterpreter(RNN_TYPE, RNN_STATE_LENGTH, NB_RNN_LAYERS, nb_cells)
print(f"{sequence_interpreter}")
sequence_interpreter.to(device)

for i in range(nb_curriculum_steps):
    pass

exit()

train_dataset = SequenceInterpreterDataset(f"{DATASET_PATH}/train.csv")
val_dataset = SequenceInterpreterDataset(f"{DATASET_PATH}/val.csv")
test_dataset = SequenceInterpreterDataset(f"{DATASET_PATH}/test.csv")

train_dataloader = DataLoader(train_dataset, BATCH_SIZE, True, drop_last=True)
val_dataloader = DataLoader(val_dataset, BATCH_SIZE, False, drop_last=True)
test_dataloader = DataLoader(test_dataset, BATCH_SIZE, False, drop_last=True)

loss_function = nn.BCEWithLogitsLoss()
optimizer = torch.optim.AdamW(sequence_interpreter.parameters(), lr=0.001, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=10)

for i in range(NB_EPOCHS):
    print(f"epoch {i + 1}")
    print(f"learning rate = {optimizer.param_groups[0]["lr"]}")
    training_loss, training_accuracy = train_the_model(train_dataloader, sequence_interpreter, loss_function, optimizer, device)
    validation_loss, validation_accuracy = evaluate_the_model(val_dataloader, sequence_interpreter, loss_function, device)
    scheduler.step(validation_loss)
    print(f"training loss = {training_loss}")
    print(f"validation loss = {validation_loss}")
    print(f"training accuracy = {training_accuracy}")
    print(f"validation accuracy = {validation_accuracy}")
    print()
