from learning import collate_fn
from learning import evaluate_the_model
from learning import SequenceInterpreter
from learning import SequenceInterpreterDataset
from learning import train_the_model
from pathlib import Path
import torch
from torch import nn
from torch.utils.data import DataLoader

ACCURACY_TARGET = 0.98
BATCH_SIZE = 64
BIDIRECTIONAL = False
DATASET_PATH = "datasets/dev"
INITIAL_LEARNING_RATE = 0.0005
MAX_GRADIENT_NORM = 1
NB_EPOCHS = 1000
NB_RNN_LAYERS = 1
RESET_LEARNING_RATE = False
RESET_SCHEDULER = True
RNN_STATE_LENGTH = 1000
RNN_TYPE = "GRU"
WEIGHT_DECAY = 0.1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"{device}")
print()

# GETTING INFO
validation_path = f"{DATASET_PATH}/val"
validation_files = sorted([str(path) for path in Path(validation_path).iterdir()])
dataset = SequenceInterpreterDataset(validation_files[0])
input_tensor, target_tensor = dataset[0]
NB_CELLS = target_tensor.shape[1]
NB_CURRICULUM_STEPS = len(validation_files)
with open(validation_files[0], "r", encoding="utf-8") as file:
    text = file.read()
    INCLUDE_INTERMEDIATE_STEPS = ":" in text
print(f"NB_CELLS = {NB_CELLS}")
print(f"NB_CURRICULUM_STEPS = {NB_CURRICULUM_STEPS}")
print(f"INCLUDE_INTERMEDIATE_STEPS = {INCLUDE_INTERMEDIATE_STEPS}")
print()

# MODEL CREATION
sequence_interpreter = SequenceInterpreter(RNN_TYPE, RNN_STATE_LENGTH, NB_RNN_LAYERS, BIDIRECTIONAL, NB_CELLS, INCLUDE_INTERMEDIATE_STEPS)
print(f"{sequence_interpreter}")
print()
sequence_interpreter.to(device)

loss_function = nn.BCEWithLogitsLoss()
optimizer = torch.optim.AdamW(sequence_interpreter.parameters(), lr=INITIAL_LEARNING_RATE, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=10)

# CURRICULUM
for i in range(NB_CURRICULUM_STEPS):

    nb_digits = len(str(NB_CURRICULUM_STEPS))
    train_dataset = SequenceInterpreterDataset(f"{DATASET_PATH}/train/{i + 1:0{nb_digits}d}.csv")
    val_dataset = SequenceInterpreterDataset(f"{DATASET_PATH}/val/{i + 1:0{nb_digits}d}.csv")

    train_dataloader = DataLoader(train_dataset, BATCH_SIZE, True, collate_fn=collate_fn, drop_last=True)
    val_dataloader = DataLoader(val_dataset, BATCH_SIZE, False, collate_fn=collate_fn, drop_last=True)

    if RESET_SCHEDULER:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=10)
    if RESET_LEARNING_RATE:
        optimizer.param_groups[0]["lr"] = INITIAL_LEARNING_RATE

    success = False

    best_epoch = 0
    best_validation_sequence_wise_accuracy = 0

    for j in range(NB_EPOCHS):
        print(f"curriculum step {i + 1} - epoch {j + 1}")
        print(f"learning rate = {optimizer.param_groups[0]["lr"]}")
        training_loss, training_cell_wise_accuracy, training_sequence_wise_accuracy, max_grad_norm, average_grad_norm = train_the_model(train_dataloader, sequence_interpreter, loss_function, optimizer, device, MAX_GRADIENT_NORM)
        validation_loss, validation_cell_wise_accuracy, validation_sequence_wise_accuracy = evaluate_the_model(val_dataloader, sequence_interpreter, loss_function, device)
        scheduler.step(validation_loss)
        if validation_sequence_wise_accuracy > best_validation_sequence_wise_accuracy:
            best_epoch = j + 1
            best_validation_sequence_wise_accuracy = validation_sequence_wise_accuracy
        print(f"maximum gradient norm = {max_grad_norm}")
        print(f"average gradient norm = {average_grad_norm}")
        print(f"training loss = {training_loss}")
        print(f"validation loss = {validation_loss}")
        print(f"training sequence-wise accuracy = {training_sequence_wise_accuracy}")
        print(f"validation sequence-wise accuracy = {validation_sequence_wise_accuracy}")
        print(f"best epoch = {best_epoch}")
        print(f"best validation sequence-wise accuracy = {best_validation_sequence_wise_accuracy}")
        print()
        if validation_sequence_wise_accuracy >= ACCURACY_TARGET:
            success = True
            break

    if not success:
        break
