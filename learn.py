from learning import SequenceInterpreterDataset

DATASET_PATH = "datasets/dev"

train_dataset = SequenceInterpreterDataset(f"{DATASET_PATH}/train.csv")
