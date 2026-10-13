from dataset_generation import generate_dataset
from interpretations import interpretation_undo
from interpretations import interpretation_8_instructions

PATH = "datasets/dev"
PARTS = (
    ("train", 80000),
    ("val", 10000),
    ("test", 10000)
)
NB_CELLS = 10
INITIALIZATION_TYPE = "constant"
WRITE_INITIALIZATION = True
INSTRUCTIONS_GENERATION_TYPE = "curriculum"
NBS_INSTRUCTIONS = list(range(10, 1001, 10))
INSTRUCTIONS_SUBSET = [2, 3, 4, 4]
INCLUDE_INTERMEDIATE_STEPS = True
INTERPRETATION = interpretation_undo

generate_dataset(PATH, PARTS, NB_CELLS, INITIALIZATION_TYPE, WRITE_INITIALIZATION, INSTRUCTIONS_GENERATION_TYPE, NBS_INSTRUCTIONS, INSTRUCTIONS_SUBSET, INCLUDE_INTERMEDIATE_STEPS, INTERPRETATION)
