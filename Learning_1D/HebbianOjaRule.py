import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from Asym_1D import AsymNetworks as AN # Asymmetrical recurrent interaction.
from NN_1D import Networks as SN # Symmetrical recurrent interaction.
from mpl_toolkits.axes_grid1 import ImageGrid


#############################################################################################################
class FfrecTimeDevelop_1D:
    """
    Only consider one input neuron.
    Given at the beginning the inputs and let the network learning over time. Approximate the ffrec over time
    and compare the tendncy with the values of derivative factor.
    When considering one input neuron, the input autocorrelation = 1 and the constants in different samples
    does not make difference for the h_t and ffrec_t values. They are only depend on Wt.
    Therefore only use 1 for the input to get the ffrec values and derivative factor.
    """
    def __init__(self, output_n, R):

        self.input_n = 1
        self.output_n = output_n
        self.R = R

        # Firstly consider the simple case of random symmetric recurrent interaction.
        self.recurrent_interaction = SN.LinearRecurrentNetwork(output_n, self.R).interaction
        self.J_eigval, self.J_eigvec = np.linalg.eigh(self.recurrent_interaction) # Eigval, Eigvec of recurrent inter.

        self.steady_inter = np.linalg.inv(np.eye(self.output_n) - self.recurrent_interaction) # (1-J)^-1

        self.trans_rec_inter = self.recurrent_interaction @ self.steady_inter # J(1-J)^-1
        self.trans_eigval, self.trans_eigvec = np.linalg.eigh(self.trans_rec_inter)

        self.feedforward_interaction = self.feedforward_interaction_asym() # Initial feedforward interaction.

    def feedforward_interaction_asym(self):
        rng = np.random.default_rng(seed = 42)
        W = rng.normal(0, 1, size = (self.output_n, self.input_n)) # (n, 1) dimensional array.
        return W

    def weight_update_euler(self, W_old, delta_t):
        W_new = W_old + delta_t * self.steady_inter @ W_old
        W_new = W_new/np.linalg.norm(W_new)
        return W_new # (n,1) dimensional array.

    def ffrec_t(self, W_t):
        # W_t (n,1) dimensional array. Normalized when calculate ffrec.
        #W_t = W_t/np.linalg.norm(W_t)
        ffrec = W_t.T @ self.recurrent_interaction @ W_t
        return ffrec


    def derivative_factor(self, W_t):
        """
        Fowrward estimation of functional derivative.
        For normalized Wt, factor = W_t.T ( J(1-J)^-1 + (1-J)^-T J ) W_t
        = 2 Wt.T J(1-J)^-1 Wt - 2 Wt.T J Wt Wt.T (1-J)^-1 Wt
        """
        factor = W_t.T @ self.trans_rec_inter.T @ W_t + W_t.T @ self.trans_rec_inter @ W_t -\
                 2 * W_t.T @ self.recurrent_interaction @ W_t @ W_t.T @ self.steady_inter @ W_t
        return factor


    def time_update(self, delta_t, T, W0=None):
        if W0 is None:
            W0 = self.feedforward_interaction

        num_step = int(T/delta_t) + 1 # Inclusive the time 0.
        t_list = np.linspace(0, T, num_step)

        # Generate the W_t develop over time.
        W = [W0/np.linalg.norm(W0)]
        for i in range(1, len(t_list)): # exclusive t=0.
            W_old = W[i-1]
            W_new = self.weight_update_euler(W_old, delta_t)
            W.append(W_new)
        W = np.array(W)  # (num_step, n_output, 1) dimensional.

        # Calculate ffrec values and derivative values.
        ffrec = np.zeros(num_step)
        derivative_factor = np.zeros(num_step)
        for i in range(num_step):
            ffrec[i] = self.ffrec_t(W[i])
            derivative_factor[i] = self.derivative_factor(W[i])

        return W, ffrec, derivative_factor


    def plot_time_update_statistics(self, repeats, delta_t, T):
        rng = np.random.default_rng(seed = 42)
        all_start_W = rng.normal(0, 1, size = (repeats, self.output_n, self.input_n))
        all_ffrec = []
        all_factor = []
        for i in range(repeats):
            update = self.time_update(delta_t, T, W0 = all_start_W[i])
            all_ffrec += update[1].tolist()
            all_factor += update[2].tolist()

        x_axis = [i for rep in range(repeats) for i in range(int(T/delta_t) + 1)]
        plt.figure(figsize=(6,5))
        sns.lineplot(x=x_axis, y=all_ffrec, label="Feedforward recurrent \n alignment")
        sns.lineplot(x=x_axis, y=all_factor, label="Derivative")
        plt.xlabel("Time units", fontsize=18)
        plt.xticks([0, 10, 20, 30, 40, 50], fontsize=15)
        plt.yticks([0, 0.4, 0.8, 1.2, 1.6], fontsize=15)
        plt.legend(fontsize=15)
        plt.savefig("F:/Downloads/fig.pdf", bbox_inches='tight')
        plt.show()

    def weight_derivative(self, delta_t, T):
        W = self.time_update(delta_t, T)[0]
        dW_dt = self.steady_inter @ W
        return W, dW_dt

    def W_projection_rec_eigvec_coeff_1D(self, Wt):
        # Sort eigvec in A in descending order corresponding to eigenvalues.
        sort_index = np.argsort(self.J_eigval)[::-1]
        A = self.J_eigvec[:, sort_index] # sorted.
        # coeff = A^-1 * Wt
        return np.linalg.inv(A) @ Wt # (n_output, 1) dimensional.

    def all_W_projection_1D(self, all_Wt, t_list):
        coeff_1D = []
        for i in range(len(t_list)):
            Wt = all_Wt[i]
            coeff_values = np.abs(self.W_projection_rec_eigvec_coeff_1D(Wt)) # Consider absolut values.
            # See the percentage of the coefficients for the first 20 eigenvectors.
            first_ten = np.sum(coeff_values[:20])/np.sum(coeff_values)
            coeff_1D.append(first_ten)
        return coeff_1D # (n_step, ) dimensional.

    def all_W_projection_statistic(self, repeats, delta_t, total_T):
        num_step = int(total_T/delta_t) + 1 # Inclusive the time 0.
        t_list = np.linspace(0, total_T, num_step)
        all_coeff = []

        rng = np.random.default_rng(seed = 42)
        all_start_W = rng.normal(0, 1, size = (repeats, self.output_n, self.input_n))
        for i in range(repeats):
            W0 = all_start_W[i]
            all_Wt = self.time_update(delta_t, total_T, W0 = W0)[0]
            # Calculate the projection coefficients.
            proj_coeff = self.all_W_projection_1D(all_Wt, t_list)
            all_coeff += proj_coeff
        # plot the statistics.
        plt.figure()
        #plt.title("Projection Ration of the first 20 eigenvectors")
        plt.xlabel("Time units", fontsize=18)
        plt.ylabel("Ratio", fontsize=18)
        # x-axis for lineplot.
        steps = [i for rep in range(repeats) for i in range(len(t_list))]
        sns.lineplot(x=steps, y=all_coeff)
        plt.xticks(fontsize=15)
        plt.yticks([0.2, 0.6, 1.0], fontsize=15)
        plt.savefig("F:/Downloads/fig.pdf", bbox_inches='tight')
        plt.show()
        return all_coeff



#############################################################################################################
if __name__ == "__main__":
    # Parameters
    delta_t = 0.1
    total_T = 5
    n = 500
    R = 0.85
    t_list = list(range(0, total_T+1, delta_t))
    initial_rec_sym_rand = SN.LinearRecurrentNetwork(n=n, R=R)

    # Commands
    time_exp_1D = FfrecTimeDevelop_1D(output_n=n, R=R)
    W0 = time_exp_1D.feedforward_interaction
    time_exp_1D.all_W_projection_statistic(50, delta_t=0.1, total_T=5)

































