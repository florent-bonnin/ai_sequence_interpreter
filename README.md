# Sequence Interpreter

**Can a recurrent neural network learn to execute a small program?**

Sequence Interpreter is a PyTorch experiment in which GRU and LSTM networks learn to simulate a deterministic machine. The machine has a circular tape of binary cells, a movable cursor, and a sequence of instructions. Given the initial tape and the instructions, the network must predict the resulting tape—either at the end of execution or after every instruction.

The project explores **algorithmic learning, recurrent memory, sequence length, and curriculum learning**. In particular, an `undo` instruction makes the task more demanding: correctly reversing an earlier operation requires information about execution history, not just the current tape.

## How the machine works

The environment consists of:

- A fixed-length, one-dimensional **binary tape** (`NB_CELLS` cells).
- A **cursor**, initially at cell `0`.
- A **program** represented by a sequence of single-digit instruction codes.
- **Wraparound addressing**: moving left from the first cell reaches the last cell, and moving right from the last reaches the first.

The implementation includes two alternative instruction sets in [`interpretations.py`](interpretations.py). **Codes `2`, `3`, and `4` have different semantics depending on the interpreter selected**; these are separate experiments, not instructions to mix together.

### Instruction set A: local operations

Interpreter: `interpretation_8_instructions`

| Code | Operation | Effect |
| --- | --- | --- |
| `2` | Move left | Move the cursor one cell left. |
| `3` | Move right | Move the cursor one cell right. |
| `4` | Flip current cell | Invert the bit at the cursor. |
| `5` | Flip three cells | Invert the left neighbor, current cell, and right neighbor. |
| `6` | AND | Replace the current bit with the AND of its two neighbors. |
| `7` | OR | Replace the current bit with the OR of its two neighbors. |
| `8` | XOR | Replace the current bit with the XOR of its two neighbors. |
| `9` | Swap neighbors | Exchange the bits immediately to the left and right of the cursor. |

All positions use circular wraparound. The AND/OR/XOR instructions modify **only the current cell**, using the bits in the two neighboring cells.

### Instruction set B: movements with undo

Interpreter: `interpretation_undo` (**selected by the current dataset-generation configuration**).

| Code | Operation | Effect |
| --- | --- | --- |
| `2` | Left and flip | Move one cell left, flip the destination bit, and record the operation. |
| `3` | Right and flip | Move one cell right, flip the destination bit, and record the operation. |
| `4` | Undo | Reverse the most recent move that has not already been undone. |

Undo uses a last-in, first-out history of movements. It reverses both the bit flip and the cursor movement. An undo on an empty history has no effect. This introduces a form of **temporally structured memory**: the tape and cursor describe the present, while undo also depends on previous actions.

#### Worked example

Consider a five-cell tape initialized to `00000` and the instructions `3, 3, 4, 2`:

| Step | Instruction | Cursor after execution | Tape after execution |
| --- | --- | --- | --- |
| Initial | — | `0` | `00000` |
| 1 | `3` — right + flip | `1` | `01000` |
| 2 | `3` — right + flip | `2` | `01100` |
| 3 | `4` — undo | `1` | `01000` |
| 4 | `2` — left + flip | `0` | `11000` |

With initial-state encoding and intermediate supervision enabled, the corresponding CSV line would be:

```text
000003342,00000:01000:01100:01000:11000
```

The left-hand side contains the **five initial tape bits followed by the instruction codes**. On the right, the colon-separated states represent the tape before each instruction and once again after the final instruction.

## Learning task and architecture

The [`SequenceInterpreter`](learning.py) model follows a deliberately simple architecture:

```text
Initial tape bits + instruction tokens
                 │
       One-hot encoding (10 values)
                 │
       GRU or LSTM (1+ layers)
                 │
           Linear layer
                 │
        N binary output logits
```

- **Input:** an integer sequence. Tape values are encoded as `0` or `1`; instruction codes are in the range `2`–`9`.
- **Recurrent backbone:** configurable GRU or LSTM, number of layers, state size, and optional bidirectionality.
- **Output:** one logit per tape cell, converted into a binary prediction using a zero-logit threshold.
- **Variable-length sequences:** batches are padded with `-1` and handled using PyTorch's packed-sequence utilities.
- **Loss:** `BCEWithLogitsLoss`, applied only to non-padded target positions.

There are two supervision modes:

1. **Final-state prediction** (`INCLUDE_INTERMEDIATE_STEPS = False`): predict only the tape after the complete program.
2. **Intermediate-state prediction** (`INCLUDE_INTERMEDIATE_STEPS = True`): predict the tape at every program boundary, including the initial tape and the final tape. The current model takes its first prediction after reading the encoded initial tape.

Intermediate supervision makes it possible to measure whether the network reproduces the entire **execution trace**, rather than only the final result.

## Curriculum learning

Programs can be hard to learn when they are long, so the dataset generator supports progressive increases in program length. In curriculum mode, each stage is associated with a maximum instruction count. Examples at that stage have randomly selected lengths between **zero and that maximum**, inclusive.

For instance, with:

```python
INSTRUCTIONS_GENERATION_TYPE = "curriculum"
NBS_INSTRUCTIONS = [10, 20, 30]
```

three curriculum stages are generated, with instruction counts of `0–10`, `0–20`, and `0–30` respectively. The model **keeps its learned weights and optimizer state** when moving between stages. Each stage ends when validation sequence-wise accuracy reaches `ACCURACY_TARGET`, or when the maximum number of epochs is reached; if a stage fails, training stops.

The generation module also supports:

| Mode | Behavior |
| --- | --- |
| `constant` | Generate one fixed instruction sequence and reuse it throughout the dataset. |
| `variable` | Independently sample each program length from `NBS_INSTRUCTIONS`. |
| `curriculum` | Create successive datasets with increasing maximum program lengths. |

The initial tape can similarly be configured as all zeros (`zero`), one randomly generated tape shared by all examples (`constant`), or a new random tape for each example (`variable`).

### Metrics

Training and validation report two complementary accuracy measures:

- **Cell-wise accuracy:** the fraction of individual target bits predicted correctly.
- **Sequence-wise accuracy:** the fraction of examples for which **every predicted target bit is correct**. With intermediate supervision, *every intermediate tape and the final tape* must match, not just the final tape.

Sequence-wise accuracy is deliberately stringent: one incorrect bit at any evaluated step makes the entire example incorrect. The training loop also prints loss, learning rate, and gradient-norm statistics.

## Repository structure

```text
ai_sequence_interpreter/
├── dataset_generation.py  # Dataset construction and curriculum stages
├── generate_dataset.py    # Dataset-generation configuration and entry point
├── interpretations.py     # Reference interpreters for both instruction sets
├── learning.py            # Dataset loader, GRU/LSTM model, training, metrics
├── learn.py               # Training configuration and curriculum loop
├── results/               # Experiment-related files
└── README.md
```

## Installation

Requires **Python 3.10+** (for `match` / `case`) and PyTorch. The only non-standard-library dependencies imported by the code are **`torch`** and **`tabulate`**.

```bash
git clone https://github.com/florent-bonnin/ai_sequence_interpreter.git
cd ai_sequence_interpreter

python -m venv .venv
source .venv/bin/activate    # Windows PowerShell: .venv\Scripts\Activate.ps1

pip install torch tabulate
```

For GPU-specific PyTorch installation instructions, consult the [official PyTorch installation selector](https://pytorch.org/get-started/locally/). Training automatically chooses CUDA when it is available, otherwise CPU.

## Running an experiment

### 1. Configure and generate the dataset

Edit the constants at the top of [`generate_dataset.py`](generate_dataset.py), then run:

```bash
python generate_dataset.py
```

The current checked-in settings specify:

| Setting | Value |
| --- | --- |
| Dataset directory | `datasets/dev` |
| Tape length | `10` cells |
| Initial tape | `constant` (random once, shared across examples) |
| Initial tape included in input | `True` |
| Interpreter | `interpretation_undo` |
| Instruction sampling | `[2, 3, 4, 4]` |
| Program generation | `curriculum` |
| Curriculum maximum lengths | `10, 20, ..., 1000` |
| Training / validation / test examples | `80,000 / 10,000 / 10,000` **per stage** |
| Intermediate-state targets | `True` |

Repeated entries in `INSTRUCTIONS_SUBSET` intentionally change sampling probabilities. For example, `[2, 3, 4, 4]` gives the undo instruction twice the probability of either movement instruction.

**Important:** these defaults generate **100 curriculum stages and 10 million examples in total**, potentially requiring substantial disk space and training time. Also, `generate_dataset()` **deletes and recreates the selected dataset directory** every time it runs.

For a quicker exploratory run, consider replacing the dataset-generation settings with:

```python
PARTS = (("train", 2000), ("val", 400), ("test", 400))
NBS_INSTRUCTIONS = [5, 10, 20]
```

Keep `WRITE_INITIALIZATION = True` when using intermediate-state supervision: the current model aligns its output time steps by assuming that the first `NB_CELLS` input tokens encode the initial tape.

The generated directory structure is:

```text
datasets/dev/
├── train/
│   ├── 1.csv
│   ├── 2.csv
│   └── ...
├── val/
│   ├── 1.csv
│   ├── 2.csv
│   └── ...
└── test/
    ├── 1.csv
    ├── 2.csv
    └── ...
```

Each file contains one stage and each row uses the format `input,target` (with multiple colon-separated target states when intermediate steps are enabled).

### 2. Configure and train a model

Edit [`learn.py`](learn.py), making sure `DATASET_PATH` points to the generated dataset. The current model configuration uses:

| Setting | Value |
| --- | --- |
| RNN type | `GRU` |
| Number of RNN layers | `1` |
| Recurrent state size | `1000` |
| Bidirectional | `False` |
| Batch size | `64` |
| Optimizer | `AdamW` |
| Initial learning rate | `0.0005` |
| Weight decay | `0.1` |
| Gradient clipping max norm | `1` |
| LR scheduler | `ReduceLROnPlateau`, factor `0.5`, patience `10` |
| Maximum epochs per stage | `100` |
| Required validation sequence-wise accuracy | `99%` |
| Reset scheduler at each stage | `True` |
| Reset learning rate at each stage | `False` |

Run:

```bash
python learn.py
```

For a smaller initial test, you can set `RNN_STATE_LENGTH = 128` and reduce `NB_EPOCHS`. This is only a lightweight configuration suggestion, **not** a guarantee that the experiment will reach its target accuracy.

## Suggested experiments

The shared problem and training pipeline make it possible to investigate several questions:

- How does performance change as programs become longer than those mastered in earlier curriculum stages?
- How do GRUs and LSTMs compare with the same state size and different numbers of recurrent layers?
- Does a variable initial tape force the network to learn a more general computation than a fixed initial tape?
- Does predicting every intermediate state help the model learn the true instruction semantics?
- How much harder is the history-dependent `undo` interpreter than local tape operations?
- What changes when the instruction subset or relative sampling frequencies are modified?

These are **experimental directions**, not claims about demonstrated results. The implementation reports measurements during training; no benchmark numbers are asserted here.

## Implementation notes and limitations

- The repository is an **experimental training project**, not a general-purpose program interpreter or a packaged library. Configuration is done by editing the scripts rather than using CLI arguments.
- The generator creates a `test` split, but **`learn.py` currently trains on `train` and evaluates on `val` only**. There is no separate test-set evaluation command.
- The training script **does not automatically save model checkpoints or training logs**. A run's best validation epoch is displayed, but its weights are not restored automatically.
- The random data generator does not set a fixed seed, so reproducing a particular dataset requires adding explicit seeding.
- Bidirectional recurrent networks are configurable; however, the current **final-state-only** forward path takes just the last direction's state and needs adjustment for bidirectional output dimensions. Also, bidirectional predictions of intermediate states can use future instructions, so they do not represent strictly causal execution.
- Long programs with dense intermediate supervision can be computationally and memory intensive. Results on the generated length ranges should not be confused with proven generalization to arbitrary-length programs.

## Purpose

Sequence Interpreter was created as a hands-on exploration of recurrent neural networks and their ability to learn structured sequential computation. Rather than asking the network to classify a static pattern, the project asks it to maintain and update an internal representation of a changing system—and, with `undo`, to use information from its history.
