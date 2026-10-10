from dataset_generation import generate_dataset

path = "datasets/dev"
parts = (
    ("train", 80000),
    ("val", 10000),
    ("test", 10000)
)
nb_cells = 100
initialization_type = "zero"
write_initialization = True
possible_nbs_moves = [100]
change_moves = True
generate_dataset(path, parts, nb_cells, initialization_type, write_initialization, possible_nbs_moves, change_moves)
