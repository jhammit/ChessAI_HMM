from hmmlearn import hmm
import numpy as np

class HMMHandler:
    def __init__(self, n_components=3, n_iter=2200, tol=1e-4):
        self.model = hmm.GaussianHMM(n_components=n_components, n_iter=n_iter, tol=tol)

    def train(self, data):
        # Check data format (n_samples, n_features)
        if len(data.shape) == 1:
            data = data.reshape(-1, 1)  # Reshape to 1 data feature
        self.model.fit(data)

    def predict(self, sequence):
        # Check data format (n_samples, n_features)
        if len(sequence.shape) == 1:
            sequence = sequence.reshape(-1, 1)  # Reshape to 1 data feature
        return self.model.predict(sequence)

    def decode(self, sequence):
        # Decode sequence to get the most likely hidden states
        if len(sequence.shape) == 1:
            sequence = sequence.reshape(-1, 1)
        return self.model.decode(sequence)
