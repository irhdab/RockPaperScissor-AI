"""
Unit tests for game logic
Tests scoring, win rules, and game mechanics
"""

import unittest
import sys
import os
import numpy as np

# Add parent directory to path to import game logic
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config


class TestGameLogic(unittest.TestCase):
    """Test cases for game logic"""

    def test_win_rules(self):
        """Test win rules logic"""
        # Rock beats Scissors
        self.assertEqual(config.WIN_RULES['rock'], 'scissor')
        
        # Paper beats Rock
        self.assertEqual(config.WIN_RULES['paper'], 'rock')
        
        # Scissors beats Paper
        self.assertEqual(config.WIN_RULES['scissor'], 'paper')

    def test_score_calculation(self):
        """Test score calculation logic"""
        # Test tie game
        player_score, bot_score = 0, 0
        new_player_score, new_bot_score = self.update_score('rock', 'rock', player_score, bot_score)
        self.assertEqual(new_player_score, 0)
        self.assertEqual(new_bot_score, 0)

        # Test player wins
        player_score, bot_score = 0, 0
        new_player_score, new_bot_score = self.update_score('rock', 'scissor', player_score, bot_score)
        self.assertEqual(new_player_score, 1)
        self.assertEqual(new_bot_score, 0)

        # Test bot wins
        player_score, bot_score = 0, 0
        new_player_score, new_bot_score = self.update_score('rock', 'paper', player_score, bot_score)
        self.assertEqual(new_player_score, 0)
        self.assertEqual(new_bot_score, 1)

    def test_all_win_combinations(self):
        """Test all possible win combinations"""
        test_cases = [
            # (player_gesture, bot_gesture, expected_player_score, expected_bot_score)
            ('rock', 'rock', 0, 0),      # Tie
            ('rock', 'paper', 0, 1),     # Bot wins
            ('rock', 'scissor', 1, 0),   # Player wins
            ('paper', 'rock', 1, 0),     # Player wins
            ('paper', 'paper', 0, 0),    # Tie
            ('paper', 'scissor', 0, 1),  # Bot wins
            ('scissor', 'rock', 0, 1),   # Bot wins
            ('scissor', 'paper', 1, 0),  # Player wins
            ('scissor', 'scissor', 0, 0), # Tie
        ]

        for player_gesture, bot_gesture, expected_player_score, expected_bot_score in test_cases:
            with self.subTest(player_gesture=player_gesture, bot_gesture=bot_gesture):
                player_score, bot_score = 0, 0
                new_player_score, new_bot_score = self.update_score(
                    player_gesture, bot_gesture, player_score, bot_score
                )
                self.assertEqual(new_player_score, expected_player_score)
                self.assertEqual(new_bot_score, expected_bot_score)

    def test_labels_encoding(self):
        """Test label encoding is correct"""
        # Test one-hot encoding
        for gesture, encoding in config.LABELS.items():
            # Should have exactly one 1.0 and rest 0.0
            self.assertEqual(sum(encoding), 1.0)
            self.assertEqual(len(encoding), 3)
            self.assertEqual(max(encoding), 1.0)
            self.assertEqual(min(encoding), 0.0)

    def test_gesture_consistency(self):
        """Test that all gestures are consistently defined"""
        # All gestures should be in both LABELS and WIN_RULES
        gestures = set(config.LABELS.keys())
        win_rule_gestures = set(config.WIN_RULES.keys())
        
        self.assertEqual(gestures, win_rule_gestures)
        self.assertEqual(len(gestures), 3)  # rock, paper, scissor

    def update_score(self, player_play, bot_play, player_score, bot_score):
        """Helper function to test score updating logic"""
        if player_play == bot_play:
            return player_score, bot_score
        elif bot_play == config.WIN_RULES[player_play]:
            return player_score + 1, bot_score
        else:
            return player_score, bot_score + 1


if __name__ == '__main__':
    unittest.main() 