from flask import Flask, render_template, jsonify, request
from tictactoe.TicTacToeGame import TicTacToeGame as Game
from tictactoe.keras.NNet import NNetWrapper as NNet
from MCTS import MCTS
from utils import *
import numpy as np
import webbrowser
import threading
import time

app = Flask(__name__)

# Initialize game and AI
g = Game(3)
nnet = NNet(g)
try:
    nnet.load_checkpoint('./pretrained_models/tictactoe/keras/', 'best-25eps-25sim-10epch.pth.tar')
    print("Model loaded successfully!")
except Exception as e:
    print(f"Warning: Could not load model: {e}")
    print("Continuing without pretrained model...")

args = dotdict({'numMCTSSims': 50, 'cpuct': 1.0})
mcts = MCTS(g, nnet, args)

# Game state storage (in production, use sessions or database)
games = {}

@app.route('/')
def index():
    return render_template('tictactoe.html')

@app.route('/new_game', methods=['POST'])
def new_game():
    """Start a new game"""
    game_id = str(time.time())
    board = g.getInitBoard()
    player = 1  # Human is player 1 (X), AI is player -1 (O)
    games[game_id] = {
        'board': board.tolist(),
        'player': player,
        'game_ended': False,
        'result': None
    }
    return jsonify({'game_id': game_id, 'board': board.tolist()})

@app.route('/make_move', methods=['POST'])
def make_move():
    """Handle player move and AI response"""
    data = request.json
    game_id = data.get('game_id')
    row = data.get('row')
    col = data.get('col')
    
    if game_id not in games:
        return jsonify({'error': 'Game not found'}), 404
    
    game_state = games[game_id]
    board = np.array(game_state['board'])
    player = game_state['player']
    
    # Check if game already ended
    if game_state['game_ended']:
        return jsonify({'error': 'Game already ended'}), 400
    
    # Convert row, col to action
    action = g.n * row + col
    
    # Validate move
    valids = g.getValidMoves(board, player)
    if valids[action] != 1:
        return jsonify({'error': 'Invalid move'}), 400
    
    # Make player move
    board, next_player = g.getNextState(board, player, action)
    
    # Check if game ended after player move
    game_ended = g.getGameEnded(board, player)
    if game_ended != 0:
        games[game_id]['board'] = board.tolist()
        games[game_id]['game_ended'] = True
        games[game_id]['result'] = game_ended
        return jsonify({
            'board': board.tolist(),
            'game_ended': True,
            'result': game_ended,
            'message': 'You win!' if game_ended == 1 else 'Draw!' if abs(game_ended) < 0.1 else 'AI wins!'
        })
    
    # AI's turn
    canonical_board = g.getCanonicalForm(board, next_player)
    action_probs = mcts.getActionProb(canonical_board, temp=0)
    ai_action = np.argmax(action_probs)
    
    # Make AI move
    board, final_player = g.getNextState(board, next_player, ai_action)
    
    # Check if game ended after AI move (from human player's perspective)
    game_ended = g.getGameEnded(board, player)  # Check from human's perspective
    result_message = None
    if game_ended != 0:
        games[game_id]['board'] = board.tolist()
        games[game_id]['game_ended'] = True
        games[game_id]['result'] = game_ended
        if game_ended == 1:
            result_message = 'You win!'
        elif game_ended == -1:
            result_message = 'AI wins!'
        else:
            result_message = 'Draw!'
    else:
        games[game_id]['board'] = board.tolist()
        games[game_id]['player'] = final_player
    
    return jsonify({
        'board': board.tolist(),
        'game_ended': game_ended != 0,
        'result': game_ended if game_ended != 0 else None,
        'message': result_message
    })

@app.route('/get_valid_moves', methods=['POST'])
def get_valid_moves():
    """Get valid moves for current board state"""
    data = request.json
    game_id = data.get('game_id')
    
    if game_id not in games:
        return jsonify({'error': 'Game not found'}), 404
    
    game_state = games[game_id]
    board = np.array(game_state['board'])
    player = game_state['player']
    
    valids = g.getValidMoves(board, player)
    valid_moves = []
    for i, valid in enumerate(valids):
        if valid == 1 and i < g.n * g.n:
            row = i // g.n
            col = i % g.n
            valid_moves.append({'row': row, 'col': col})
    
    return jsonify({'valid_moves': valid_moves})

def open_browser():
    """Open browser after a short delay"""
    time.sleep(1.5)
    webbrowser.open('http://127.0.0.1:5000')

if __name__ == '__main__':
    # Open browser in a separate thread
    threading.Thread(target=open_browser).start()
    print("Starting TicTacToe web server...")
    print("The game will open in your browser automatically.")
    app.run(debug=False, use_reloader=False)

