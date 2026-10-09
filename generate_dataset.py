from dataset_generation import generate_dataset

path = "datasets/dev"
parts = (
    ("train", 800),
    ("val", 100),
    ("test", 100)
)
nb_cells = 10
initialization_type = 0
write_initialization = True
possible_nb_moves = [10]
change_moves = True
generate_dataset(path, parts, nb_cells, initialization_type, write_initialization, possible_nb_moves, change_moves)
