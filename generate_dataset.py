from dataset_generation import generate_dataset

path = "datasets/dev"
parts = (
    ("train", 800),
    ("val", 100),
    ("test", 100)
)
nb_cells = 100
initialization_type = "zero"
write_initialization = False
moves_type = "curriculum"
nbs_moves = list(range(200))

generate_dataset(path, parts, nb_cells, initialization_type, write_initialization, moves_type, nbs_moves)
