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

def generate_moves(nbs_moves):
    nb_moves = nbs_moves[random.randint(0, len(nbs_moves) - 1)]
    moves = []
    for i in range(nb_moves):
        moves.append(random.randint(2, 3))
    return moves

def execute_sequence(cells, moves, include_intermediate_steps):
    position = 0
    intermediate_steps = []
    for move in moves:
        if include_intermediate_steps:
            intermediate_step = [str(cell) for cell in cells]
            intermediate_step = "".join(intermediate_step)
            intermediate_steps.append(intermediate_step)
        if move == 2:
            position -= 1
            if position < 0:
                position = len(cells) - 1
        else:
            position += 1
            if position >= len(cells):
                position = 0
        cells[position] = 1 - cells[position]
    return intermediate_steps

def generate_curriculum_steps(moves_type, nbs_moves):
    curriculum_steps = []
    if moves_type == "constant":
        curriculum_step = None
        curriculum_steps.append(curriculum_step)
    elif moves_type == "variable":
        curriculum_step = nbs_moves
        curriculum_steps.append(curriculum_step)
    elif moves_type == "curriculum":
        nbs_moves = sorted(set(nbs_moves))
        for nb_moves in nbs_moves:
            curriculum_step = list(range(nb_moves + 1))
            curriculum_steps.append(curriculum_step)
    return curriculum_steps

def generate_dataset(path, parts, nb_cells, initialization_type, write_initialization, moves_type, nbs_moves, include_intermediate_steps):
    """
    initialization_type must be "zero", "constant" or "variable"
    moves_type must be "constant", "variable" or "curriculum"
    """

    if os.path.exists(path):
        shutil.rmtree(path)
    Path(path).mkdir(parents=True)

    if initialization_type in ["zero", "constant"]:
        if initialization_type == "zero":
            random_initialization = False
        else:
            random_initialization = True
        initial_cells = create_cells(nb_cells, random_initialization)

    if moves_type == "constant":
        moves = generate_moves(nbs_moves)
    curriculum_steps = generate_curriculum_steps(moves_type, nbs_moves)
    
    for part_name, part_size in parts:

        part_path = f"{path}/{part_name}"
        Path(part_path).mkdir()

        for i, curriculum_step in enumerate(curriculum_steps):
            print(f"generating curriculum step {i + 1}")
            file_name = f"{part_path}/{i + 1:0{len(str(len(curriculum_steps)))}d}.csv"
            with open(file_name, "w", encoding="utf-8") as file:
                for i in range(part_size):

                    if initialization_type in ["zero", "constant"]:
                        cells = initial_cells.copy()
                    elif initialization_type == "variable":
                        cells = create_cells(nb_cells, True)
                    if write_initialization:
                        initialization_str = "".join([str(cell) for cell in cells])
                        file.write(initialization_str)

                    if moves_type in ["variable", "curriculum"]:
                        moves = generate_moves(curriculum_step)
                    moves_str = "".join([str(move) for move in moves])
                    file.write(moves_str)

                    file.write(",")

                    intermediate_steps = execute_sequence(cells, moves, include_intermediate_steps)
                    target_str = "".join([str(cell) for cell in cells])
                    intermediate_steps.append(target_str)
                    intermediate_steps = ":".join(intermediate_steps)
                    file.write(intermediate_steps)

                    file.write("\n")
