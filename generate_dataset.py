from dataset_generation import generate_dataset

path = "datasets/dev"
parts = (
    ("train", 80000),
    ("val", 10000),
    ("test", 10000)
)
nb_cells = 10
initialization_type = "zero"
write_initialization = True
moves_type = "curriculum"
nbs_moves = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
include_intermediate_steps = True

generate_dataset(path, parts, nb_cells, initialization_type, write_initialization, moves_type, nbs_moves, include_intermediate_steps)
