# ChessAI_HMM

# ChessAI with Probabilistic Learning (HMM)

This project extends the original ChessAI project by "markusbuffett" to include probabilistic learning using Hidden Markov Models (HMM). The original project focused on various search methods, board setup, and gameplay, while this extension introduces predictive elements to the Chess AI through HMM.

## Overview

The original ChessAI project, developed by "markusbuffett", provided foundational components for a chess-playing AI, including:
- Search algorithms for move selection
- Board representation and setup
- Basic gameplay mechanics

### New Additions

This extension introduces probabilistic learning techniques via HMM in two main areas:
- **AI.py**: Modifications to integrate HMM-based predictions into the decision-making process of the AI. This allows the AI to anticipate opponent moves probabilistically rather than relying solely on a move tree.
- **hmmlearn.py**: Implementation of the Hidden Markov Model for learning opponent behaviors and making informed moves based on predicted outcomes.

## Usage

To use the extended ChessAI with HMM:
1. Clone the repository.
2. Install any necessary dependencies.
3. Run the AI using the provided scripts or integrate it into your own chess application.


## Credits

- Original ChessAI project by "markusbuffett": https://github.com/marcusbuffett/command-line-chess
- HMM implementation and extensions by JHammit

## License

This project is licensed under the [MIT License](LICENSE).

