from dataset_generation import generate_dataset

path = "datasets/dev"
parts = (
    ("train", 800),
    ("val", 100),
    ("test", 100)
)
nb_cells = 10
nb_moves = 0
initialization_type = 2
write_initialization = False
generate_dataset(path, parts, nb_cells, nb_moves, initialization_type, write_initialization)
