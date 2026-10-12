def keep_in_space(nb_cells, position):
    if position == -1:
        position = nb_cells - 1
    elif position == nb_cells:
        position = 0
    return position

def interpretation_8_instructions(cells, instructions, include_intermediate_steps):
    nb_cells = len(cells)
    position = 0
    intermediate_steps = []
    for instruction in instructions:
        if include_intermediate_steps:
            intermediate_step = [str(cell) for cell in cells]
            intermediate_step = "".join(intermediate_step)
            intermediate_steps.append(intermediate_step)
        match instruction:
            case 2:
                position = keep_in_space(nb_cells, position - 1)
            case 3:
                position = keep_in_space(nb_cells, position + 1)
            case 4:
                cells[position] = 1 - cells[position]
            case 5:
                shifts = [-1, 0, 1]
                for shift in shifts:
                    neighbor = keep_in_space(nb_cells, position + shift)
                    cells[neighbor] = 1 - cells[neighbor]
            case 6:
                left = keep_in_space(nb_cells, position - 1)
                right = keep_in_space(nb_cells, position + 1)
                cells[position] = int(cells[left] and cells[right])
            case 7:
                left = keep_in_space(nb_cells, position - 1)
                right = keep_in_space(nb_cells, position + 1)
                cells[position] = int(cells[left] or cells[right])
            case 8:
                left = keep_in_space(nb_cells, position - 1)
                right = keep_in_space(nb_cells, position + 1)
                cells[position] = int(cells[left] != cells[right])
            case 9:
                left = keep_in_space(nb_cells, position - 1)
                right = keep_in_space(nb_cells, position + 1)
                t = cells[left]
                cells[left] = cells[right]
                cells[right] = t
    return intermediate_steps

def interpretation_undo(cells, instructions, include_intermediate_steps):
    nb_cells = len(cells)
    position = 0
    intermediate_steps = []
    history = []
    for instruction in instructions:
        if include_intermediate_steps:
            intermediate_step = [str(cell) for cell in cells]
            intermediate_step = "".join(intermediate_step)
            intermediate_steps.append(intermediate_step)
        match instruction:
            case 2:
                position = keep_in_space(nb_cells, position - 1)
                cells[position] = 1 - cells[position]
                history.append(instruction)
            case 3:
                position = keep_in_space(nb_cells, position + 1)
                cells[position] = 1 - cells[position]
                history.append(instruction)
            case 4:
                if history:
                    instruction_to_undo = history.pop()
                    match instruction_to_undo:
                        case 2:
                            cells[position] = 1 - cells[position]
                            position = keep_in_space(nb_cells, position + 1)
                        case 3:
                            cells[position] = 1 - cells[position]
                            position = keep_in_space(nb_cells, position - 1)
    return intermediate_steps
