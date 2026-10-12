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

def generate_instructions(nbs_instructions, instructions_subset):
    nb_instructions = nbs_instructions[random.randint(0, len(nbs_instructions) - 1)]
    instructions = []
    for i in range(nb_instructions):
        instruction = random.choice(instructions_subset)
        instructions.append(instruction)
    return instructions

def keep_in_space(nb_cells, position):
    if position == -1:
        position = nb_cells - 1
    elif position == nb_cells:
        position = 0
    return position

def generate_curriculum_steps(instructions_generation_type, nbs_instructions):
    curriculum_steps = []
    if instructions_generation_type == "constant":
        curriculum_step = None
        curriculum_steps.append(curriculum_step)
    elif instructions_generation_type == "variable":
        curriculum_step = nbs_instructions
        curriculum_steps.append(curriculum_step)
    elif instructions_generation_type == "curriculum":
        nbs_instructions = sorted(set(nbs_instructions))
        for nb_instructions in nbs_instructions:
            curriculum_step = list(range(nb_instructions + 1))
            curriculum_steps.append(curriculum_step)
    return curriculum_steps

def generate_dataset(path, parts, nb_cells, initialization_type, write_initialization, instructions_generation_type, nbs_instructions, instructions_subset, include_intermediate_steps, interpretation):
    """
    initialization_type must be "zero", "constant" or "variable"
    instructions_generation_type must be "constant", "variable" or "curriculum"
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

    if instructions_generation_type == "constant":
        instructions = generate_instructions(nbs_instructions, instructions_subset)
    curriculum_steps = generate_curriculum_steps(instructions_generation_type, nbs_instructions)
    
    for part_name, part_size in parts:

        part_path = f"{path}/{part_name}"
        Path(part_path).mkdir()

        for i, curriculum_step in enumerate(curriculum_steps):
            if instructions_generation_type == "curriculum":
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

                    if instructions_generation_type in ["variable", "curriculum"]:
                        instructions = generate_instructions(curriculum_step, instructions_subset)
                    instructions_str = "".join([str(instruction) for instruction in instructions])
                    file.write(instructions_str)

                    file.write(",")

                    intermediate_steps = interpretation(cells, instructions, include_intermediate_steps)
                    target_str = "".join([str(cell) for cell in cells])
                    intermediate_steps.append(target_str)
                    intermediate_steps = ":".join(intermediate_steps)
                    file.write(intermediate_steps)

                    file.write("\n")
