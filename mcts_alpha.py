import numpy as np
import math
import torch

class Node:
    def __init__(self, game, args, state, parent=None, action_taken=None, prior=0):
        """
        Initializes a node in the MCTS tree.
        """
        self.game = game
        self.args = args
        self.state = state
        self.parent = parent
        self.action_taken = action_taken
        self.prior = prior
        
        self.children = []  # List to store child nodes
        
        self.visit_count = 0
        self.value_sum = 0  # Tracks cumulative value of the node

    def is_fully_expanded(self):
        """
        Returns True if there are children.
        """
        return len(self.children) > 0
    
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

        The UCB score balances exploitation (using known good moves) with exploration (trying new moves).
        It combines three terms:
        1. Q-value: The average reward (converted from [-1,1] to [0,1] range)
        2. Visit count ratio: Encourages exploration of less-visited nodes
        3. Prior probability: The policy network's initial evaluation of the move

        Args:
            child (Node): The child node to compute the UCB score for

        Returns:
            float: The UCB score for the child node

        Formula:
            UCB = Q + C * sqrt(N) / (n + 1) * P
            where:
            Q = transformed average value 
            C = exploration constant
            N = parent visit count
            n = child visit count
            P = prior probability
        """
        if child.visit_count == 0:
            q_value = 0
        else:
            # The average reward (convert from [-1,1] to [0,1] range)
            q_value = 1 - ((child.value_sum / child.visit_count) + 1) / 2
        return q_value + self.args['C'] * (math.sqrt(self.visit_count) / (child.visit_count + 1)) * child.prior
    
    def expand(self, policy):
        """
        Expands the current node by creating child nodes for each valid action.
        
        Args:
            policy (list): Probabilities for each action (0 for invalid actions).
        """
        for action, prob in enumerate(policy):
            # Only actions with probability > 0 are expanded
            if prob > 0:
                # copy current state and apply the action
                child_state = self.state.copy()
                child_state = self.game.get_next_state(child_state, action, 1)
                # Perspective is changed to opponent's view (-1) for each child state
                child_state = self.game.change_perspective(child_state, player=-1)

                # add child node
                self.children.append(
                    Node(self.game, self.args, child_state, self, action, prob)
                )

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
    def __init__(self, game, args, model):
        """
        Initializes the Monte Carlo Tree Search (MCTS) algorithm.
        """
        self.game = game
        self.args = args
        self.model = model

    @torch.no_grad()
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
                policy, value = self.model(
                    torch.tensor(self.game.get_encoded_state(node.state)).unsqueeze(0)
                )
                policy = torch.softmax(policy, axis=1).squeeze(0).cpu().numpy()
                valid_moves = self.game.get_valid_moves(node.state)
                policy *= valid_moves
                policy /= np.sum(policy)
                
                value = value.item()
                
                node.expand(policy)
                
            node.backpropagate(value)
            
        action_probs = np.zeros(self.game.action_size)
        for child in root.children:
            action_probs[child.action_taken] = child.visit_count
        action_probs /= np.sum(action_probs)
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