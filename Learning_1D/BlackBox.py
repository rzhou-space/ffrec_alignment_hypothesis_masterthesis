import matplotlib.pyplot as plt
import matplotlib.colors as cm
import numpy as np
import seaborn as sns
#from Asym_1D import AsymStillWork
from sklearn.decomposition import PCA

class BlackBox:

    def __init__(self, n, R, network):
        self.neuron = n
        self.R = R
        self.interaction = network.interaction # Unknown interaction matrix. Depends on the network structure.

        rng = np.random.default_rng(seed = 42)
        self.steady_inter = np.linalg.inv(np.eye(self.neuron) - self.interaction) # (I-J)^-1

        # Construct random orthonomal basis.
        # Firstly generate a random square matrix.
        rand_matrix = rng.normal(0, 1, size=(self.neuron, self.neuron)) # --> have faster learning.
        # rand_matrix = rng.random(size = (self.neuron, self.neuron)) # uniform distributed --> slower learning.
        # Apply the gram-schmit to orthogonalize the random matrix.
        Q,R = np.linalg.qr(rand_matrix)
        # The columns are the basis vectors.
        self.orthobasis = Q


    def sigma_spont(self, kappa, beta_spont, basis_vec, L=1):
        """
        Generate the covariance matrix with low dimensional input (M = kappa * beta)
        dimensional. The contruction was applied in the alignment to spontaneous activity.
        """
        # Determine the upper limit factor M.
        M = kappa * beta_spont
        # Calculate the input variance Sigma^spont.
        sigma_spont = np.zeros((self.neuron, self.neuron))
        for i in range(L-1, L-1+M):
            v_i = np.exp(-2*(i-L)/beta_spont)
            # Getting vectors in columns as basis vector and transform it into nx1 dimensional numpy array.
            e_i = basis_vec[:, i].reshape(-1,1)
            sigma_spont += v_i * (e_i @ e_i.T) # Should be a nxn dimensional numpy matrix.
        return sigma_spont


    def n_th_response(self, n_turn, cov_sigma, num_sample):
        # The covariance matrix for the n-th response after n-turns of applying prior repsonses as inputs.
        # ((1-J)^-1)^n+1 * sigma_spont * ((1-J)^-T)^n+1
        n_cov = np.linalg.matrix_power(self.steady_inter, n_turn+1) @ cov_sigma \
                @ np.linalg.matrix_power(self.steady_inter.T, n_turn+1)
        # Generate the responses.
        rng = np.random.default_rng(seed = 42)
        n_response = rng.multivariate_normal(mean = np.zeros(self.neuron),
                                             cov = n_cov, size = num_sample)
        # Normalize the n_responses row-wise, i.e., each response sample would be normalized. -- Think not necessary.
        row_norms = np.linalg.norm(n_response, axis=1)
        n_response = n_response / row_norms[:, np.newaxis]
        return n_response, n_cov # (n_sample, n_neuron) dimensional array.


    def align_ffrec(self, n_turn, cov_sigma, num_sample):
        # Calculate the ffrec for responses after applying n-1 times the prior responses.
        n_input = self.n_th_response(n_turn-1, cov_sigma, num_sample)[0]  # With normalized rows.
        # For each row, calculate the ffrec. n_row = num_sample.
        ffrec = np.zeros(num_sample)
        for i in range(num_sample):
            input = n_input[i]
            ffrec[i] = input @ self.interaction @ input
        return ffrec  # (num_sample, ) dimensional


    def align_ffrec_PCA(self, n_turn, cov_sigma, num_sample):
        '''
        # Calculate the ffrec for responses after applying n times the prior responses with the PC.
        n_responses = self.n_th_response(n_turn, cov_sigma, num_sample)[0].T # (n_neuron, n_sample) dimenaional.
        # Do PCA on the n_resposes.
        pca = PCA(n_components=self.neuron)
        PCs = pca.fit_transform(n_responses)
        variance_ratio = pca.explained_variance_ratio_
        '''
        n_cov = self.n_th_response(n_turn, cov_sigma, num_sample)[1]
        cov_eigval, cov_eigvec = np.linalg.eigh(n_cov) # Covariance matrix symmetrical.
        # Sort eigenvectors (PCs) and eigenvalues (variance ratio) in descending order.
        sort_index = np.argsort(cov_eigval)[::-1]
        variance_ratio = cov_eigval[sort_index]
        variance_ratio = variance_ratio/np.linalg.norm(variance_ratio)
        PCs = cov_eigvec[:, sort_index]

        ffrec = np.zeros(len(variance_ratio))
        for i in range(len(variance_ratio)):
            h = PCs[:, i]/np.linalg.norm(PCs[:, i])
            ffrec[i] = h @ self.interaction @ h
        return variance_ratio, ffrec


    # Functions for mean_ffrec plot against steps for multiple dimensionas.
    def mean_ffrec_step(self, num_step, kappa, beta_dim, num_sample):
        """
        Calculate the mean ffrec for each step under a certain dimensionality.
        """
        mean_ffrec = []
        cov_sigma = self.sigma_spont(kappa, beta_dim, self.orthobasis)
        for n_turn in range(num_step):
            mean_ffrec.append(np.mean(self.align_ffrec(n_turn, cov_sigma, num_sample)))
        return mean_ffrec # (num_step, ) dimensional.


    def mean_ffrec_step_multi_dim(self, num_step, kappa, beta_list, num_sample):
        '''
        Calcualte for multiple dimensions the mean ffrec over num_samples.
        '''
        mean_ffrec_multi_dim = []
        for dim in beta_list:
            mean_ffrec_multi_dim.append(self.mean_ffrec_step(num_step, kappa, dim, num_sample))
        return np.array(mean_ffrec_multi_dim) # (num_dim, num_step) dimensional.


    def plot_mean_ffrec_step_multi_dim(self, num_step, kappa, beta_list, num_sample):
        """
        Plot the mean_ffrec curves against steps for different dimensions.
        """
        mean_ffrecs = self.mean_ffrec_step_multi_dim(num_step, kappa, beta_list, num_sample) # Get (num_dim, num_step) dimensional array.
        plt.figure()
        for i in range(len(beta_list)):
            plt.plot(mean_ffrecs[i], label="dim = " + str(beta_list[i]))
            plt.xticks(range(0,num_step))
        plt.xlabel("step")
        plt.ylabel("mean ffrec")
        plt.legend()
        plt.show()


    def plot_ffrec_step_multi_dim(self, num_step, kappa, beta_list, num_sample):
        """
        development of ffrec alignment against the times of applying previous output as input.
        """
        plt.figure(figsize=(6,5))
        plt.xlabel("Step", fontsize=18)
        plt.ylabel("Feedforward recurrent alignment", fontsize=18)
        for beta_dim in beta_list:
            cov_sigma = self.sigma_spont(kappa, beta_dim, self.orthobasis)
            all_ffrec = [] # Stores at each step num_sample of ffrec value.
            x_step = [] # x-axis value for each value in all_ffrec.
            for n_turn in range(1, num_step):
                # Final length of all_ffrec, x_step would be num_step*num_sample.
                all_ffrec += self.align_ffrec(n_turn, cov_sigma, num_sample).tolist()
                x_step += [n_turn] * num_sample # Repeat n_turn for num_step times.
            # Lineplot with error interval.
            sns.lineplot(x=x_step, y=all_ffrec) #, label="dim = "+ str(beta_dim))
            plt.xticks([1, 20, 40, 60, 80, 100], fontsize=15)
            plt.yticks([0.7, 0.76, 0.82], fontsize=15)
        #plt.legend()
        plt.savefig("F:/Downloads/fig.pdf", bbox_inches='tight')
        plt.show()



############################################################################################################
if __name__ == "__main__":
    # paramters
    n_neuron = 200
    R = 0.85
    kappa = 5
    beta_spont = 20
    beta_dim = 10
    num_sample = 500

    from NN_1D import Networks
    from Asym_1D import AsymNetworks
    network_sym = Networks.LinearRecurrentNetwork(n_neuron, R)
    network_asym = AsymNetworks.AsymLinearRecurrentNet(n_neuron, R)
    test = BlackBox(n_neuron, R, network_sym) # Model for symmetric networks.
    test_asym = BlackBox(n_neuron, R, network_asym) # Model for asymmetric networks.

    # Commands...






