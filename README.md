# AlphaZero TicTacToe

A simplified implementation of AlphaZero algorithm for playing TicTacToe. This project uses self-play reinforcement learning to train an AI that can play TicTacToe.

## 🚀 Setup

### Prerequisites

- Python 3.7 or higher
- pip (Python package manager)

### Installation Steps

1. **Clone or download this repository**

2. **Create a virtual environment (Recommended)**

   **Windows:**
   ```bash
   python -m venv venv
   venv\Scripts\activate
   ```

   **Linux/Mac:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

   This will install:
   - TensorFlow/Keras (for neural networks)
   - NumPy, tqdm, and other dependencies
   - Flask (for web interface)

## 🎮 Running the Game

### Option 1: Web Interface (Recommended)

Play TicTacToe in your browser with a beautiful UI:

```bash
python tictactoe_web.py
```

This will:
- Start a web server on `http://127.0.0.1:5000`
- Automatically open your browser
- Load a pretrained TicTacToe model (if available)
- Let you play against the AI

**Note:** If the model file is missing, the game will still work but the AI will use an untrained network.

### Option 2: Command Line Interface

Play via command line:

```bash
python pit.py
```

Edit `pit.py` to configure:
- Which player you want to be (human or AI)
- Opponent type (random player, AI, or human)

## 🎓 Training Your Own Model

To train a new TicTacToe AI model:



1. **Configure training parameters** in `main.py`:

   ```python
   args = dotdict({
       'numIters': 100,              # Number of training iterations
       'numEps': 100,                 # Self-play games per iteration
       'numMCTSSims': 25,             # MCTS simulations per move
       'arenaCompare': 40,            # Games to compare new vs old model
       'updateThreshold': 0.6,        # Win rate threshold to accept new model
       'maxlenOfQueue': 200000,       # Max training examples
       'checkpoint': './temp/',       # Save directory
       'load_model': False,           # Load existing model
   })
   ```

2. **Start training**:

   ```bash
   python main.py
   ```

   The training process will:
   - Play self-play games
   - Collect training data
   - Train the neural network
   - Evaluate and save the best model

   Trained models will be saved in the `./temp/` directory.

## 📁 Project Structure

```
alpha-zero-general/
├── main.py                 # Training script
├── pit.py                  # Play games via command line
├── tictactoe_web.py        # Web interface
├── Coach.py                # Training loop
├── MCTS.py                 # Monte Carlo Tree Search
├── Arena.py                # Game arena
├── tictactoe/              # TicTacToe game implementation
│   ├── TicTacToeGame.py    # Game logic
│   ├── TicTacToeLogic.py   # Board logic
│   ├── TicTacToePlayers.py # Player implementations
│   └── keras/              # Neural network
│       └── NNet.py         # Network wrapper
├── templates/              # Web templates
│   └── tictactoe.html      # Web UI
└── pretrained_models/      # Pretrained models
    └── tictactoe/
```

## ⚙️ Configuration

### Training Parameters

Key parameters you can adjust in `main.py`:

- `numIters`: Number of training iterations (more = better but slower)
- `numEps`: Self-play games per iteration (more = better training data)
- `numMCTSSims`: MCTS simulations per move (more = stronger AI but slower)
- `updateThreshold`: Win rate needed to accept new model (0.6 = 60%)

### Neural Network Parameters

Edit `tictactoe/keras/NNet.py` to adjust:

- `lr`: Learning rate (default: 0.001)
- `epochs`: Training epochs per iteration (default: 10)
- `batch_size`: Batch size for training (default: 64)

## 🐛 Troubleshooting

### Model Loading Error

If you see "No model in path" error:
- The pretrained model might be missing
- The game will still work with an untrained network
- To use a trained model, train one using `main.py` or download a pretrained model

### Web Server Won't Start

- Check if port 5000 is already in use
- Ensure Flask is installed: `pip install flask`
- Try manually opening `http://127.0.0.1:5000` in your browser

### Training Issues

**Out of Memory:**
- Reduce `batch_size` in `tictactoe/keras/NNet.py`
- Reduce `numEps` in `main.py`

**Training Too Slow:**
- Reduce `numMCTSSims`
- Use a GPU if available
- Reduce `numEps`

---

**Enjoy playing TicTacToe! 🎉**
