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
