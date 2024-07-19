from __future__ import annotations
import random
import numpy as np
from hmmlearn import hmm
from src.Board import Board
from src.InputParser import InputParser
from src.Move import Move
from src.MoveNode import MoveNode

WHITE = True
BLACK = False

class AI:
    depth = 1
    movesAnalyzed = 0

    class HMMHandler:
        def __init__(self, n_components: int = 1):  # Reduced components to avoid degenerate solution (previously 3)
            self.model = hmm.GaussianHMM(n_components=n_components, covariance_type='diag', n_iter=150)

        def train(self, data: np.ndarray) -> None:
            # Ensure data is formatted (n_samples, n_features)
            if len(data.shape) == 1:
                data = data.reshape(-1, 1)
            self.model.fit(data)

        def predict(self, sequence: np.ndarray) -> np.ndarray:
            # Ensure data is formatted (n_samples, n_features)
            if len(sequence.shape) == 1:
                sequence = sequence.reshape(-1, 1)
            return self.model.predict(sequence)

    def __init__(self, board: Board, side: bool, depth: int):
        self.board = board
        self.side = side
        self.depth = depth
        self.parser = InputParser(self.board, self.side)
        self.hmm_handler = self.HMMHandler()
        self.train_hmm_model()

    def train_hmm_model(self) -> None:
        # Sample data for training should be replaced with real-world data if available
        data = np.array([
            [1],
            [2],
            [3]
        ])
        self.hmm_handler.train(data)

    def predict_opponent_move(self, recent_moves: np.ndarray) -> int:
        return self.hmm_handler.predict(recent_moves.reshape(-1, 1))[-1]

    def get_recent_moves(self) -> np.ndarray:
        # Get recent moves from the board, this is example
        return np.array([
            [1],
            [2],
            [3],
            [4],
            [5]
        ])

    def getRandomMove(self) -> Move:
        legalMoves = list(self.board.getAllMovesLegal(self.side))
        return random.choice(legalMoves)

    def generateMoveTree(self) -> list[MoveNode]:
        moveTree = []
        for move in self.board.getAllMovesLegal(self.side):
            moveTree.append(MoveNode(move, [], None))

        for node in moveTree:
            self.board.makeMove(node.move)
            self.populateNodeChildren(node)
            self.board.undoLastMove()
        return moveTree

    def populateNodeChildren(self, node: MoveNode) -> None:
        node.pointAdvantage = self.board.getPointAdvantageOfSide(self.side)
        node.depth = node.getDepth()
        if node.depth == self.depth:
            return

        side = self.board.currentSide

        legalMoves = self.board.getAllMovesLegal(side)
        if not legalMoves:
            if self.board.isCheckmate():
                node.move.checkmate = True
                return
            elif self.board.isStalemate():
                node.move.stalemate = True
                node.pointAdvantage = 0
                return
            raise Exception()

        for move in legalMoves:
            self.movesAnalyzed += 1
            child_node = MoveNode(move, [], node)
            node.children.append(child_node)
            self.board.makeMove(move)
            self.populateNodeChildren(child_node)
            self.board.undoLastMove()

    def getOptimalPointAdvantageForNode(self, node: MoveNode) -> int:
        if node.children:
            for child in node.children:
                child.pointAdvantage = self.getOptimalPointAdvantageForNode(
                    child,
                )

            # If the depth is divisible by 2,
            # it's a move for the AI's side, so return max
            if node.children[0].depth % 2 == 1:
                return max(node.children).pointAdvantage
            else:
                return min(node.children).pointAdvantage
        else:
            return node.pointAdvantage

    def bestMovesWithMoveTree(self, moveTree: list[MoveNode]) -> list[Move]:
        bestMoveNodes: list[MoveNode] = []
        for moveNode in moveTree:
            moveNode.pointAdvantage = self.getOptimalPointAdvantageForNode(moveNode)
            if not bestMoveNodes:
                bestMoveNodes.append(moveNode)
            elif moveNode > bestMoveNodes[0]:  # Comparison of MoveNode objects
                bestMoveNodes = []
                bestMoveNodes.append(moveNode)
            elif moveNode == bestMoveNodes[0]:
                bestMoveNodes.append(moveNode)

        return [node.move for node in bestMoveNodes]

    def getBestMove(self) -> Move:
        recent_moves = self.get_recent_moves()
        predicted_move = self.predict_opponent_move(recent_moves)
        moveTree = self.generateMoveTree()
        bestMoves = self.bestMovesWithMoveTree(moveTree)

        # Optionally filter moves based on prediction
        best_moves = [move for move in bestMoves if self.is_move_based_on_prediction(move, predicted_move)]

        if not best_moves:
            return self.getRandomMove()  # Fallback to random move if predicted moves are no-go

        randomBestMove = random.choice(best_moves)
        randomBestMove.notation = self.parser.notationForMove(randomBestMove)
        return randomBestMove

    def is_move_based_on_prediction(self, move: Move, predicted_move: int) -> bool:
        # See if move is predicted or random
        return True

    def makeBestMove(self) -> None:
        self.board.makeMove(self.getBestMove())

if __name__ == '__main__':
    mainBoard = Board()
    ai = AI(mainBoard, True, 3)
    print(mainBoard)
    ai.makeBestMove()
    print(mainBoard)
    print(ai.movesAnalyzed)
    print(mainBoard.movesMade)
