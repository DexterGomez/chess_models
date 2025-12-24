# Chess MLX CNN

This project contains an implementation of a CNN model that plays Chess.

It is based on Alpha Zero architecture but trained using supervised learning.

## Usage (to play chess)

### Streamlit interface

Create a Python enviroment and install dependencies from `requierements.txt`.

Then run `streamlit run app.py`

### UCI Engine

Universal chess Interface (UCI) is an open communication protocol that enables chess engines to communicate with user interfaces. By using a client BanksiaGUI, you can enable an engine with the model so you, or another chess engine can play against this model.

Follow this steps to enable the UCI engine.

* First you need to create a python environment and install the dependencies from the file `requirements.txt`
* Open `run_engine.sh` file on a text editor and change the activation commands for the python enviroment to your own python enviroment.
* Save the changes and go to the client, engines, and add engine. You should specify to use the `run_engine.sh` file as client. 

### Aditional notes

The project uses MLX so there may be incompatibilties or bugs with other OS (Windows/Linux) or CPU architectures (x86_64).

## Dataset

### Data sources

Data processed by [Lichess Elite Database](https://lichess.org/team/lichess-elite-database) team was extensively used, they provide a 'curated' version of data from [Lichess](https://database.lichess.org) which is an open database for games. Their processing includes filtering for matches with players with +2200 rating.

You can find and download the database [here](https://database.nikonoel.fr).

The matches data is stored as **Portable Game Notation (PGN)** files which are text plain format that can be read by human and by software. Check wikipedia to [learn more about PGN](https://en.wikipedia.org/wiki/Portable_Game_Notation).

The script `src.download_data.py` was generated to download the data and unzip it. As to the date of when the script was made, the total storage size of the uncompressed data is around 32GB.

### Data pre-processing

To train the model the data needs to be converted in a model readible format, it completely depends on the type and model architecture to set the transformartions required. Because of the architecture of the model, we require extensive transformations as described from here.

The transformations required produce the following output format.

#### Summary

Each match should be processed as a series of board states (and a movement for it). Based on that, the dataset should be encoded as *N* board/movement states, but each state cannon be represented as a single 8*8 board, so the next encoding is used:

| Key | Shape | Dtype | Description |
| :---- | :---- | :---- | :---- |
| `x` | `(N, 14, 8, 8)` | `float32` | Game board representation |
| `policy` | `(N,)` | `int64` | Movement target (0 to 4095). |
| `value` | `(N,)` | `float32` | Result target (-1.0 to 1.0). |

#### Key tensor `x`

This tensor represents the chess board with the following informantion:

* 14 channels
* 8 board columns
* 8 board rows

The channels encode the following positional and game informantion information:

* `0 - 5`: Players's pieces (Pawn, Knight, Bishop, Rook, Queen, King)
* `5 - 11`: Opponent's pieces (Pawn, Knight, Bishop, Rook, Queen, King)
* `12`: Castling, whether or not the player is allowed to do it (1.0 yes, 0.0 no)
* `13`: 'En Passant': 1.0 on the capture target square, 0.0 otherwise

The board is encoded in 'cannonical perspective', the dataset is transformed to train the model as if it nevers plays black, for this we rotate the board 180 degrees to make the model think it is playing as white whenever it plays as black. The justification for this is to increase the efficiency of the training (reducing the amount of movements/decisions the model need to generalize). Think of it like the model needs to learn "what is the best move for the player currently at the bottom of the board?" rather than "what is the best move for white/black?".

#### Key `policy`

This encodes the movement the player does at the current state of the board.

This uses a mapping of 'from-to' movements as a flaten representation that ranges from 0 to 4095. It works with the following formula;

`index = (origin * 64) + destination`

The `origin` and `destination` are asigned by setting numerical values to each of the 64 squares of the grid (`a1=0`, `b1=0`, `a2=8`... `h8=63`) and then using the formula to get the index.

If the board is rotated, the index must be calculated using the new perspective, not the original index from the board state.

#### Key `value`

Represents the result overall result of the match from the current board state perspective:

* `+1.0` player won the match.
* `-1.0` player lost the match.
* `0.0` draw.

## Model architecture

The core of this project is the class `ChessNet`, a CNN implemnted with MLX which follows the archiecture design AlphaZero simplified. It consists of a *dual-head design* that uses a single network for the *policy* (movements) and *value* (chances to win), and a *residual tower* that relies on residual blocks that allow the network to be deeper without suffering from gradient vanishing.

The network has tree main stages:

### Input representation

The model accepts an input tensor of the shape `(N,14,8,8)` following PyTorch `NCHW` convention ([see more](https://docs.pytorch.org/tutorials/intermediate/memory_format_tutorial.html)), representing the board and movement at a given board state as described in the 'pre-processing' section.

NOTE: The `ChessNet` class handles the conversion from `NCHW` to `NHWC` convention to comply with MLX's operations requirements.

### Backbone

The backbone transforms the raw board representation into abstract feature maps.
* Initial Convolution Block:
  * Expands the 14 input channels to 128 filters.
  * Kernel size: 3×3, Padding: 1, Stride: 1.
  * Followed by Batch Normalization and ReLU activation.
* Residual Tower:
  * Consists of 5 stacked Residual Blocks (ResBlock).
  * Each block contains two 3×3 convolutional layers (128 filters) with Batch Normalization.
  * Skip Connections: The input of the block is added to the output of the second convolution before the final activation.

### Output heads

After the residual tower, the network splits into two separate "heads."

#### Policy Head (Move Selection)

Predicts the best move to play from the current position.
* Convolution: Reduces depth from 128 to 32 filters (1×1 kernel).
* Flatten: Reshapes feature maps into a vector.
* Fully Connected: Projects to 4096 outputs.
* Interpretation: The output represents logits for move probabilities (covering a simplified 64×64 move space).

#### Value Head (Position Evaluation)

Estimates the game outcome from the current player's perspective.

* Convolution: Reduces depth from 128 to 3 filters (1×1 kernel).
* Dense Layer 1: Projects flattened features to a hidden layer of size 64 (ReLU activated).
* Dense Layer 2: Projects to a single scalar output.
* Activation: Tanh (Hyperbolic Tangent), constraining the final output to the range [-1, 1].

## Model training

The model was trained using supervised learning over a database of around 4.2 million chess positions from around 400k matches (sampled ramdomly).

The training pipeline was implemented using Apple MLX to get most out of the unified memory archiecture of a M1 Max 32GB RAM setup.

### Parameters

| Parameter | Value | Description |
| :---- | :---- | :---- |
| Optimizer | adam | learning rate: 1e-3 |
| Batch size | 2048 | around 18GB memory usage |
| Epochs | 10 | 11 hours training |
| Loss function | custom | CrossEntropy (policy) + MSE (value) |

### Results

| Epoch | Avg Total Loss | Policy Loss (Logits) | Value Loss (MSE) |
| :---- | :---- | :---- |  :---- |
| 1  | 3.5181 | 2.6720 | 0.8461
| 3  | 2.5667 | 1.7526 | 0.8141
| 5  | 2.3746 | 1.5705 | 0.8041
| 8  | 2.2069 | 1.4096 | 0.7974
| 10 | 2.1308 | 1.3409 | 0.7900

Training information show low improvement in Value Loss, which would lead to a low perfomance in determining player's chances to win or lose. In real scenarios it appears to corrently understand 'obvious' win/lose scenarios. For policy performance, the model has been optimized quite well and is able to decide to move the right piece in most escenarios.

More formal evaluations and benchmarking is pending.

The model's poor loss performance may be caused by the changes made to reduce model size compared to Alpha Zero. This could be specifically for backbone severly undersized and value head without enough layers to have the depth required to handle the complexity of the board state.

## References

### Research & Architecture
* **AlphaZero:** Silver, D., et al. (2017). [Mastering the Game of Go without Human Knowledge](https://doi.org/10.1038/nature24270)
* **ResNets:** He, K., et al. (2016). [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385).

### Data
* **Lichess Elite Database:** [Standard Rated Games](https://database.nikonoel.fr) curated by the Lichess Elite Database team.
* **Lichess:** [Open Database](https://database.lichess.org).

### Tools
* **Apple MLX:** [MLX Framework](https://github.com/ml-explore/mlx) for machine learning on Apple Silicon.
* **UCI Protocol:** [Universal Chess Interface Specification](https://www.wbec-ridderkerk.nl/html/UCIProtocol.html).

## Author notes

The decision to use Apple's MLX instead of PyTorch raises from the need to get the most out of the available hardware. You can 'translate' the code to PyTorch easly as they are similar, however you would require some tricks (out of my knowledge) to reuse the weights (if even possible).

My next steps may include intensive use of RL or a new model architeture, we will see.