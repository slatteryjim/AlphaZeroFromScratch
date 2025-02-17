import numpy as np
import math

class Node:
    def __init__(self, game, args, state, parent=None, action_taken=None):
        """
        Initializes a node in the MCTS tree.
        """
        self.game = game
        self.args = args
        self.state = state
        self.parent = parent
        self.action_taken = action_taken
        
        self.children = []  # List to store child nodes
        self.expandable_moves = game.get_valid_moves(state)  # List of valid moves that can still be expanded
        
        self.visit_count = 0
        self.value_sum = 0  # Tracks cumulative value of the node

    def is_fully_expanded(self):
        """
        Returns True if all valid moves have been expanded and there are children.
        """
        return np.sum(self.expandable_moves) == 0 and len(self.children) > 0

    def select(self):
        """
        Selects the child node with the highest UCB value.
        """
        best_child = None
        best_ucb = -np.inf

        for child in self.children:
            ucb = self.get_ucb(child)
            if ucb > best_ucb:
                best_child = child
                best_ucb = ucb

        return best_child

    def get_ucb(self, child):
        """
        Computes the Upper Confidence Bound (UCB) score for a child node.
        """
        q_value = 1 - ((child.value_sum / child.visit_count) + 1) / 2
        return q_value + self.args['C'] * math.sqrt(math.log(self.visit_count) / child.visit_count)

    def expand(self):
        """
        Expands the node by selecting an untried valid move and creating a new child node.
        """
        action = np.random.choice(np.where(self.expandable_moves == 1)[0])
        self.expandable_moves[action] = 0  # Mark action as expanded

        child_state = self.state.copy()
        child_state = self.game.get_next_state(child_state, action, 1)
        child_state = self.game.change_perspective(child_state, player=-1)

        child = Node(self.game, self.args, child_state, self, action)
        self.children.append(child)
        return child

    def simulate(self):
        """
        Runs a rollout simulation from the current state to estimate its value.
        """
        value, is_terminal = self.game.get_value_and_terminated(self.state, self.action_taken)
        value = self.game.get_opponent_value(value)

        if is_terminal:
            return value

        rollout_state = self.state.copy()
        rollout_player = 1
        while True:
            valid_moves = self.game.get_valid_moves(rollout_state)
            action = np.random.choice(np.where(valid_moves == 1)[0])
            rollout_state = self.game.get_next_state(rollout_state, action, rollout_player)
            value, is_terminal = self.game.get_value_and_terminated(rollout_state, action)
            if is_terminal:
                if rollout_player == -1:
                    value = self.game.get_opponent_value(value)
                return value    

            rollout_player = self.game.get_opponent(rollout_player)

    def backpropagate(self, value):
        """
        Propagates the simulation result back up the tree, updating visit counts and value sums.
        """
        self.value_sum += value
        self.visit_count += 1

        value = self.game.get_opponent_value(value)
        if self.parent is not None:
            self.parent.backpropagate(value)

class MCTS:
    def __init__(self, game, args,
                 add_dirichlet_noise=True,
                 dirichlet_epsilon=0.25,
                 dirichlet_alpha=0.03):
        """
        Initializes the Monte Carlo Tree Search (MCTS) algorithm.
        """
        self.game = game
        self.args = args
        self.add_dirichlet_noise = add_dirichlet_noise
        self.dirichlet_epsilon   = dirichlet_epsilon
        self.dirichlet_alpha     = dirichlet_alpha

    def search(self, state):
        """
        Runs MCTS search and returns action probabilities for the root state.
        """
        root = Node(self.game, self.args, state)

        for search in range(self.args['num_searches']):
            node = root

            # Traverse the tree until a node that is not fully expanded is found
            while node.is_fully_expanded():
                node = node.select()  # Select the best child node (UCB score)

            # Determine the value of the node
            value, is_terminal = self.game.get_value_and_terminated(node.state, node.action_taken)
            value = self.game.get_opponent_value(value)

            if not is_terminal:
                node = node.expand()
                value = node.simulate()

            node.backpropagate(value)    

        # Compute action probabilities based on visit counts of child nodes
        action_probs = np.zeros(self.game.action_size)
        for child in root.children:
            action_probs[child.action_taken] = child.visit_count
        
        # Add Dirichlet noise to the root node's action probabilities
        if self.add_dirichlet_noise:
            # Get noise parameters with defaults (e.g., epsilon=0.25, alpha=0.03)
            epsilon = self.dirichlet_epsilon
            alpha   = self.dirichlet_alpha
            
            # Identify valid moves at the root
            valid_moves = self.game.get_valid_moves(state)
            valid_indices = np.where(valid_moves == 1)[0]
            if len(valid_indices) > 0:
                # Generate Dirichlet noise over valid moves
                noise = np.random.dirichlet([alpha] * len(valid_indices))
                # Mix the visit counts with the noise
                for i, move in enumerate(valid_indices):
                    action_probs[move] = (1 - epsilon) * action_probs[move] + epsilon * noise[i]
        
        # Normalize the action probabilities
        total = np.sum(action_probs)
        if total > 0:
            action_probs /= total
        return action_probs
    
    def ensemble_searches(self, state, num_searches):
        """
        Runs multiple MCTS searches sequentially and averages the action probabilities.
        """

        # Run multiple MCTS searches sequentially and average the action probabilities
        results = [self.search(state) for _ in range(num_searches)]

        # Average the action probabilities from all searches
        avg_action_probs = np.mean(results, axis=0)
        return avg_action_probs
