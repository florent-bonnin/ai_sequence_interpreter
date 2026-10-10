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
moves_type = "curriculum"
nbs_moves = list(range(100 + 1))

generate_dataset(path, parts, nb_cells, initialization_type, write_initialization, moves_type, nbs_moves)
