import numpy as np
import matplotlib.pyplot as plt

class LinearRecurrentNetwork:

    def __init__(self, n, R):
        """
        :param n: The number of neurons.
        :param R: The eigenvalue radius.
        """
        self.neuron = n
        self.R = R
        # Applying random interaction matrix.
        self.interaction = self.random_interaction_matrix()


    def random_interaction_matrix(self):
        """
        Generating the random interaction matrix (nxn dimensional) with normal distribution.
        The matrix is real, full rank and symmetric.
        :return: The interaction matrix.
        """
        #mu = 0
        #sigma = self.R/np.sqrt(self.neuron)
        rng = np.random.default_rng(seed = 42)
        J = rng.normal(0, 1, size = (self.neuron,self.neuron))
        # J should be a symmetric matrix.
        J = (J + J.T)/2
        #normalize by largest eigenvalue
        eigvals = np.linalg.eigvals(J)
        J = J*self.R/np.max(eigvals)
        return J

    def eigval_distribution(self):
        J = self.interaction
        eigval = np.linalg.eigvals(J)
        plt.scatter(eigval, np.zeros(len(eigval)), s=1)
        plt.show()



class LowRank:

    def __init__(self, n, D, R):
        self.neuron = n
        self.rank = D  # The rank of the interaction matrix.
        self.R = R  # The eigenvalue normalization factor.
        self.interaction = self.low_rank_inter() # (n,n) dimenaional symmatrical.

    def low_rank_inter(self):
        # Generate the basis vector that construct the matrix.
        rng = np.random.default_rng(seed = 42)
        m = rng.normal(0, 1, size = (self.neuron, self.rank)) # The columns are basis vectors.
        # Orthogonalization of m.
        m,r = np.linalg.qr(m) # m is a Matrix with orthonormal columns.
        inter_J = np.zeros((self.neuron, self.neuron))
        for i in range(self.rank):
            m_i = m[:, i].reshape(-1, 1) # get (n_neuron, 1) dimensional vector.
            # m_i @ m_i^T symmetrical. inter_J is sum of symmetrical matrices.
            inter_J += (1/self.neuron)*(m_i @ m_i.T)
        # Normalize the interaction matrix by R.
        eigvals = np.linalg.eigvalsh(inter_J)
        inter_J = inter_J*self.R/np.max(eigvals)
        return inter_J

    def eigval_distribution(self):
        eigvals = np.linalg.eigvalsh(self.interaction) # Eigenvalues are real numbers.
        plt.figure(figsize=(6,5))
        plt.scatter(eigvals, np.zeros(len(eigvals)))
        plt.yticks([])
        plt.xticks([0, self.R], fontsize=15)
        plt.savefig("F:/Downloads/fig.pdf", bbox_inches='tight')
        plt.show()
        #return eigvals



class NoisedLowRank_1D:
    """
    Construct the low-rank matrix with rank 1 and random noise.
    """
    def __init__(self, n, R):
        self.neuron = n
        self.R = R
        self.interaction = self.noise_low_rank_inter()

    def noise_low_rank_inter(self):
        # Generate the low rank matrix basis.
        rng = np.random.default_rng(seed = 42)
        m = rng.normal(0, 1, size = self.neuron) # (self.neuron, ) dimensional.
        # Normalization of m and turn it into (self.neuron, 1) dimensional.
        m = m/np.linalg.norm(m)
        m = m.reshape((self.neuron, 1))
        # The noise/random symmetircal part of the matrix.
        rand_J = rng.normal(0, 1, size = (self.neuron, self.neuron))
        rand_J = (rand_J + rand_J.T)/2
        # Construct the low rank interaction as low rank part + random part.
        inter_J = (m @ m.T)/self.neuron + rand_J # Symmetrical (n_neuron, n_neuron) matrix.
        # Normalization of the matrix.
        eigval = np.linalg.eigvalsh(inter_J)
        inter_J = inter_J * self.R / np.max(eigval)
        return inter_J

    def eigval_distriution(self):
        eigvals = np.linalg.eigvalsh(self.interaction)
        plt.figure()
        plt.scatter(eigvals, np.zeros(len(eigvals)))
        plt.show()
        return eigvals



class NoisedLowRank_nD:
    """
    Construct low rank interaction matrix with rank n (which shluld be significantly
    smaller than number of neurons)
    """

    def __init__(self, n, R, D):
        self.neuron = n
        self.R = R
        self.rank = D
        self.interaction = self.noise_low_rank_inter()

    def noise_low_rank_inter(self):
        # Generate the random basis vector set.
        rng = np.random.default_rng(seed=42)
        m = rng.normal(0, 1, size = (self.neuron, self.rank))
        # Orthogonalization of m.
        q, r = np.linalg.qr(m)
        m = q # Matrix with orthonormal columns.
        low_J = np.zeros((self.neuron, self.neuron))
        for i in range(self.rank):
            m_i = m[:, i].reshape(-1, 1)  # Turn the (n_neuron,) dimension into (n_neuron, 1) dimension.
            low_J += (1/self.neuron)*(m_i @ m_i.T)
        # The noise/random symmetircal part of the matrix.
        rand_J = rng.normal(0, 1, size = (self.neuron, self.neuron))
        rand_J = (rand_J + rand_J.T)/2
        # Construct the low rank interaction as low rank part + random part.
        inter_J = low_J + rand_J
        # Normalization of the matrix.
        eigval = np.linalg.eigvalsh(inter_J)
        inter_J = inter_J * self.R / np.max(eigval)
        return inter_J

    def eigval_distriution(self):
        eigvals = np.linalg.eigvalsh(self.interaction)
        plt.figure()
        plt.scatter(eigvals, np.zeros(len(eigvals)))
        plt.show()
        return eigvals


######################################################################################################
if __name__ == "__main__":
    # Global setting.
    n_neuron = 200
    R = 0.85
    sigma_trial = 0.02
    N_trial = 100

    # Commands...













