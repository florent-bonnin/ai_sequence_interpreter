import os
from pathlib import Path
import random
import shutil

def create_cells(nb_cells, random_initialization):
    cells = []
    for i in range(nb_cells):
        if random_initialization:
            cell = random.randint(0, 1)
        else:
            cell = 0
        cells.append(cell)
    return cells

def generate_dataset(path, parts, nb_cells, nb_moves, initialization_type, write_initialization):

    if os.path.exists(path):
        shutil.rmtree(path)
    Path(path).mkdir(parents=True)

    if initialization_type in [0, 1]:
        if initialization_type == 0:
            random_initialization = False
        else:
            random_initialization = True
        initial_cells = create_cells(nb_cells, random)
    for part_name, part_size in parts:
        file_name = f"{path}/{part_name}.csv"
        with open(file_name, "w", encoding="utf-8") as file:
            for i in range(part_size):
                if initialization_type in [0, 1]:
                    cells = initial_cells.copy()
                elif initialization_type == 2:
                    cells = create_cells(nb_cells, True)
                if write_initialization:
                    initialization = "".join([str(cell) for cell in cells])
                    file.write(initialization)
                file.write("\n")
