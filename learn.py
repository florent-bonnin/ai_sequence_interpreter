from learning import collate_fn
from learning import evaluate_the_model
from learning import SequenceInterpreter
from learning import SequenceInterpreterDataset
from learning import train_the_model
from pathlib import Path
import torch
from torch import nn
from torch.utils.data import DataLoader

BATCH_SIZE = 64
BIDIRECTIONAL = False
DATASET_PATH = "datasets/dev"
NB_EPOCHS = 1000
NB_RNN_LAYERS = 2
RNN_STATE_LENGTH = 1000
RNN_TYPE = "GRU"
WEIGHT_DECAY = 0.1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"{device}")
print()

# GETTING INFO
train_path = f"{DATASET_PATH}/train"
train_files = sorted([str(path) for path in Path(train_path).iterdir()])
nb_curriculum_steps = len(train_files)
dataset = SequenceInterpreterDataset(train_files[0])
input_tensor, target_tensor = dataset[0]
nb_cells = target_tensor.shape[0]

# MODEL CREATION
sequence_interpreter = SequenceInterpreter(RNN_TYPE, RNN_STATE_LENGTH, NB_RNN_LAYERS, BIDIRECTIONAL, nb_cells)
print(f"{sequence_interpreter}")
print()
sequence_interpreter.to(device)

loss_function = nn.BCEWithLogitsLoss()
optimizer = torch.optim.AdamW(sequence_interpreter.parameters(), lr=0.001, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=10)

# CURRICULUM
for i in range(nb_curriculum_steps):

    nb_digits = len(str(nb_curriculum_steps))
    train_dataset = SequenceInterpreterDataset(f"{DATASET_PATH}/train/{i + 1:0{nb_digits}d}.csv")
    val_dataset = SequenceInterpreterDataset(f"{DATASET_PATH}/val/{i + 1:0{nb_digits}d}.csv")

    train_dataloader = DataLoader(train_dataset, BATCH_SIZE, True, collate_fn=collate_fn, drop_last=True)
    val_dataloader = DataLoader(val_dataset, BATCH_SIZE, False, collate_fn=collate_fn, drop_last=True)

    for j in range(NB_EPOCHS):
        print(f"curriculum step {i + 1} - epoch {j + 1}")
        print(f"learning rate = {optimizer.param_groups[0]["lr"]}")
        training_loss, training_accuracy = train_the_model(train_dataloader, sequence_interpreter, loss_function, optimizer, device)
        validation_loss, validation_accuracy = evaluate_the_model(val_dataloader, sequence_interpreter, loss_function, device)
        scheduler.step(validation_loss)
        print(f"training loss = {training_loss}")
        print(f"validation loss = {validation_loss}")
        print(f"training accuracy = {training_accuracy}")
        print(f"validation accuracy = {validation_accuracy}")
        print()
        if validation_accuracy == 1 and training_accuracy == 1:
            break
