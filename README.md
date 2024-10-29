# Connect 4 AI with Deep Learning and MCTS

A Connect 4 AI implementation using Deep Learning and Monte Carlo Tree Search (MCTS), inspired by AlphaGo Zero's approach. The AI learns through self-play and uses a combination of neural networks and tree search for decision making.

## Features

- **Deep Neural Network Architecture**
  - Dual-headed network (policy and value heads)
  - Convolutional layers for pattern recognition
  - Policy head for move probabilities
  - Value head for position evaluation

- **Monte Carlo Tree Search**
  - UCB1-based node selection
  - Neural network guidance
  - Adaptive exploration with temperature parameter
  - State caching for efficiency

- **Training Pipeline**
  - Self-play data generation
  - Experience replay
  - TensorBoard integration
  - Progressive model evaluation

- **Interactive Interface**
  - Pygame-based GUI
  - Human vs AI mode
  - AI vs AI demonstration mode
  - Training progress visualization

# Installatoin

Clone, pip install -r requirements.txt and run python main.py