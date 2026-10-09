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

def generate_moves(possible_nb_moves):
    nb_moves = possible_nb_moves[random.randint(0, len(possible_nb_moves) - 1)]
    moves = []
    for i in range(nb_moves):
        moves.append(random.randint(0, 1))
    return moves

def execute_sequence(cells, moves):
    position = 0
    for move in moves:
        if move == 0:
            position -= 1
            if position < 0:
                position = len(cells) - 1
        else:
            position += 1
            if position >= len(cells):
                position = 0
        cells[position] = 1 - cells[position]

def generate_dataset(path, parts, nb_cells, initialization_type, write_initialization, possible_nb_moves, change_moves):

    if os.path.exists(path):
        shutil.rmtree(path)
    Path(path).mkdir(parents=True)

    if initialization_type in [0, 1]:
        if initialization_type == 0:
            random_initialization = False
        else:
            random_initialization = True
        initial_cells = create_cells(nb_cells, random_initialization)
    
    if not change_moves:
        moves = generate_moves(possible_nb_moves)

    for part_name, part_size in parts:
        file_name = f"{path}/{part_name}.csv"
        with open(file_name, "w", encoding="utf-8") as file:
            for i in range(part_size):

                if initialization_type in [0, 1]:
                    cells = initial_cells.copy()
                elif initialization_type == 2:
                    cells = create_cells(nb_cells, True)
                if write_initialization:
                    initialization_str = "".join([str(cell) for cell in cells])
                    file.write(initialization_str)

                if change_moves:
                    moves = generate_moves(possible_nb_moves)
                moves_str = "".join([str(move) for move in moves])
                file.write(moves_str)

                file.write(",")

                execute_sequence(cells, moves)
                target_str = "".join([str(cell) for cell in cells])
                file.write(target_str)

                file.write("\n")
