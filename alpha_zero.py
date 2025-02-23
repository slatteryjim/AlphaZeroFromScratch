import numpy as np
import torch
from tqdm import trange
from mcts_classic import MCTS
from datetime import datetime
from pathlib import Path


class AlphaZero:
    """
    AlphaZero implementation for self-play training of game-playing AI.
    This class implements the AlphaZero algorithm which combines Monte Carlo Tree Search (MCTS)
    with deep neural networks to learn board game strategies through self-play.
    Args:
        model: Neural network model that predicts action probabilities and state values
        optimizer: Optimizer used to train the neural network
        game: Game environment that implements required interface for state/action handling
        args: Dictionary containing hyperparameters and configuration settings:
            - num_iterations: Number of total training iterations
            - num_selfPlay_iterations: Number of self-play games per iteration
            - num_epochs: Number of training epochs per iteration
    """
    def __init__(self, model, optimizer, game, args):
        self.model = model
        self.optimizer = optimizer
        self.game = game
        self.args = args
        self.mcts = MCTS(game, args, model)
        
    def selfPlay(self):
        """
        Executes one episode of self-play and returns the game history.
        """
        memory = []
        player = 1
        state = self.game.get_initial_state()
        
        while True:
            neutral_state = self.game.change_perspective(state, player)
            action_probs = self.mcts.search(neutral_state)
            
            memory.append((neutral_state, action_probs, player))
            
            action = np.random.choice(self.game.action_size, p=action_probs)
            
            state = self.game.get_next_state(state, action, player)
            
            value, is_terminal = self.game.get_value_and_terminated(state, action)
            
            if is_terminal:
                returnMemory = []
                for hist_neutral_state, hist_action_probs, hist_player in memory:
                    hist_outcome = value if hist_player == player else self.game.get_opponent_value(value)
                    returnMemory.append((
                        self.game.get_encoded_state(hist_neutral_state),
                        hist_action_probs,
                        hist_outcome
                    ))
                return returnMemory
            
            player = self.game.get_opponent(player)
    
    def train(self, memory):
        """
        Trains the neural network on stored game memory.
        """
        pass
    
    def learn(self):
        """
        Main training loop that performs several iterations, alternating between
        self-play data generation and neural network training.
        Saves model checkpoints and other training data in organized directories.
        """
        # Create directory structure for this training run
        runs_dir = Path("runs")
        runs_dir.mkdir(exist_ok=True)

        run_dir = runs_dir / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        run_dir.mkdir(exist_ok=True)

        checkpoints_dir = run_dir / "checkpoints"
        checkpoints_dir.mkdir(exist_ok=True)

        for iteration in trange(self.args['num_iterations']):

            # Generate self-play data
            memory = []
            self.model.eval()
            for selfPlay_iteration in trange(self.args['num_selfPlay_iterations'], desc=f"Self-Play"):
                memory += self.selfPlay()
            
            # Train neural network on these self-play results
            self.model.train()
            for epoch in trange(self.args['num_epochs'], desc=f"Training epochs"):
                self.train(memory)

            # Save checkpoints
            torch.save(self.model.state_dict(), 
                      checkpoints_dir / f"iter_{iteration + 1:03d}_model.pt")
            torch.save(self.optimizer.state_dict(), 
                      checkpoints_dir / f"iter_{iteration + 1:03d}_optimizer.pt")
