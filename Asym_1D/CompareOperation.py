import matplotlib.pyplot as plt
import matplotlib.colors as cm
import seaborn as sns
import numpy as np
import AsymNetworks as AN
import scipy as sp
from sklearn.decomposition import PCA
from NN_1D import NWoperations
import AsymOperations as AO
from NN_1D import Networks as SN  # Symmetrical networks.
# For trying out... import the Still Work module.
import AsymStillWork as ASW



# Trial to trial correlation comparisom of different interaction matrix.
#############################################################################################################
# Asymmetrical networks that could be taken into account.

def compare_ttc_only_combi_inter(n_neuron, R, sigma_trial, N_trial, mode):
    """
    Compare the trial to trial correlation with J = a*J_sym + (1-a)*J_asym of varias a value.
    """
    a_values = np.linspace(0, 1, 5)
    colors = [cm.to_hex(plt.cm.tab20b(i)) for i in range(20)]
    plt.figure(figsize=(6, 5))
    for i in range(len(a_values)):
        a = a_values[i]
        network = AN.CombiAsymNet(n_neuron, a, R)
        ttc_class = AO.TrialtoTrialCor(n_neuron, R, network)
        if mode == "real part":
            results = ttc_class.real_ttc_sort_align(sigma_trial, N_trial)
            ffrec = results[0]
            ttc = results[1]
            plt.scatter(ffrec, ttc, c=colors[i], alpha=0.5, label="a="+str(a))
            plt.xticks([-0.5, 0, 0.5], fontsize=15)
            plt.yticks([0, 0.5, 1], fontsize=15)
        elif mode == "symmetrized":
            results = ttc_class.sym_ttc_sort_align(sigma_trial, N_trial)
            ffrec = results[0]
            ttc = results[1]
            plt.scatter(ffrec, ttc, c=colors[i], alpha=0.5, label="a="+str(a))
            plt.xticks([-1, -0.5, 0, 0.5, 1], fontsize=15)
            plt.yticks([0, 0.5, 1], fontsize=15)
        elif mode == "white noise":
            results = ttc_class.noise_ttc_sort_align(sigma_trial, N_trial)
            ffrec = results[0]
            ttc = results[1]
            plt.scatter(ffrec, ttc, c=colors[i], alpha=0.5, label="a="+str(a))
            plt.xlabel("Feedforward recurrent alignment", fontsize=18)
            plt.ylabel("Trial-to-trial correlation", fontsize=18)
            plt.xticks([-1, -0.5, 0, 0.5, 1], fontsize=15)
            plt.yticks([0, 0.5, 1], fontsize=15)

    #plt.legend(bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left",
    #            mode="expand", borderaxespad=0, ncol=5, fontsize=15) # Make legend bar outside the figure.
    plt.legend(fontsize=15) # Normal legend bar inside figure.
    plt.savefig("F:/Downloads/fig.pdf", bbox_inches='tight') # Save figure as pdf.
    plt.show()


def compare_ttc_low_rank(n_neuron, R, sigma_trial, N_trial, mode, sigma_1=1, sigma_2=1):
    # Variation on the rank of the interaction.
    rank_values = [1, 5]
    #colors = [cm.to_hex(plt.cm.tab20b(i)) for i in range(20)] # Colors that not differ much.
    colors = [cm.to_hex(plt.cm.tab10(i)) for i in range(10)]
    plt.figure(figsize=(6,5))
    for i in range(len(rank_values)):
        D = rank_values[i]
        # Different networks setup:
        #network = AN.LowRank(n_neuron, D, R, sigma_1, sigma_2) # Asymmetrical low rank without noise.
        #network = SN.NoisedLowRank_1D(n_neuron, R) # Symmetrical noised low rank.
        network = AN.NoisedLowRank(n_neuron, D, R) # Asymmetric with noise.
        ttc_class = AO.TrialtoTrialCor(n_neuron, R, network)
        if mode == "real part":
            results = ttc_class.real_ttc_sort_align(sigma_trial, N_trial)
            ffrec = results[0]
            ttc = results[1]
            plt.scatter(ffrec, ttc, c=colors[i], alpha=0.5, label="rank G ="+str(D))
            #sns.lineplot(x=ffrec, y=ttc, label="rank="+str(D))
        elif mode == "symmetrized":
            results = ttc_class.sym_ttc_sort_align(sigma_trial, N_trial)
            ffrec = results[0]
            ttc = results[1]
            plt.scatter(ffrec, ttc, c=colors[i], alpha=0.5, label="rank G ="+str(D))
            plt.xlabel("Feedforward recurrent alignment", fontsize=18)
            plt.ylabel("Trial-to-trial correlation", fontsize=18)
            plt.xticks([-1, -0.5, 0, 0.5, 1], fontsize=15)
            plt.yticks([0, 0.5, 1], fontsize=15)
        elif mode == "white noise":
            results = ttc_class.noise_ttc_sort_align(sigma_trial, N_trial)
            ffrec = results[0]
            ttc = results[1]
            plt.scatter(ffrec, ttc, c=colors[i], alpha=0.5, label="rank="+str(D))

    plt.xlabel("Feedforward recurrent alignment", fontsize=20)
    plt.ylabel("Trial-to-trial correlation", fontsize=20)
    plt.legend()
    #plt.legend(bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left", mode="expand", ncol=5 ,  fontsize=15)
    plt.savefig("F:/Downloads/fig.pdf", bbox_inches='tight')
    plt.show()


# Intra Trial Stability
##############################################################################################################
def compare_its_only_combi_inter(n_neuron, R, dt_euler, dt_intra, T, sigma_time, mode):
    """
    Compare the intra trial correlation with J = a*J_sym + (1-a)*J_asym of varias a value.
    """
    a_values = np.linspace(0, 1, 5)
    colors = [cm.to_hex(plt.cm.tab20b(i)) for i in range(20)]
    plt.figure(figsize=(6, 5))
    for i in range(len(a_values)):
        a = a_values[i]
        network = AN.CombiAsymNet(n_neuron, a, R)
        its_class = AO.IntraTrialStab(n_neuron, R, network)
        if mode == "real part":
            results = its_class.plot_real_sort_stab(dt_euler, dt_intra, T, sigma_time)
            ffrec = results[0]
            its = results[1]
            plt.scatter(ffrec, its, c=colors[i], alpha=0.5, label="a="+str(a))
            plt.xticks([-0.5, 0, 0.5], fontsize=15)
            plt.yticks([0, 0.4, 0.8], fontsize=15)
        elif mode == "symmetrized":
            results = its_class.sym_sort_stab(dt_euler, dt_intra, T, sigma_time)
            ffrec = results[0]
            its = results[1]
            plt.scatter(ffrec, its, c=colors[i], alpha=0.5, label="a="+str(a))
            plt.xticks([-1, -0.5, 0, 0.5, 1], fontsize=15)
            plt.yticks([0, 0.4, 0.8], fontsize=15)
        elif mode == "white noise":
            results = its_class.noise_sort_stab(dt_euler, dt_intra, T, sigma_time)
            ffrec = results[0]
            its = results[1]
            plt.scatter(ffrec, its, c=colors[i], alpha=0.5, label="a="+str(a))
            plt.xlabel("Feedforward recurrent alignment", fontsize=18)
            plt.ylabel("Intra-trial stability", fontsize=18)
            plt.xticks([-1, -0.5, 0, 0.5, 1], fontsize=15)
            plt.yticks([0, 0.4, 0.8], fontsize=15)

    plt.legend(fontsize=15)
    plt.savefig("F:/Downloads/fig_its.pdf", bbox_inches='tight')
    plt.show()


def compare_its_low_rank(n_neuron, R, dt_euler, dt_intra, T, sigma_time, mode, sigma_1=1, sigma_2=1):
    # Variation on the rank of the interaction.
    rank_values = [1,5]
    #colors = [cm.to_hex(plt.cm.tab20b(i)) for i in range(20)]
    colors = [cm.to_hex(plt.cm.tab10(i)) for i in range(10)]
    plt.figure(figsize=(6,5))
    for i in range(len(rank_values)):
        D = rank_values[i]
        # Different network setup:
        #network = AN.LowRank(n_neuron, D, R, sigma_1, sigma_2)
        network = AN.NoisedLowRank(n_neuron, D, R)
        its_class = AO.IntraTrialStab(n_neuron, R, network)
        if mode == "real part":
            results = its_class.plot_real_sort_stab(dt_euler, dt_intra, T, sigma_time)
            ffrec = results[0]
            its = results[1]
            plt.scatter(ffrec, its, c=colors[i], alpha=0.5, label="rank="+str(D))
        elif mode == "symmetrized":
            results = its_class.sym_sort_stab(dt_euler, dt_intra, T, sigma_time)
            ffrec = results[0]
            its = results[1]
            plt.scatter(ffrec, its, c=colors[i], alpha=0.5, label="rank="+str(D))
            plt.xlabel("Feedforward recurrent alignment", fontsize=18)
            plt.ylabel("Intra-trial stability", fontsize=18)
            plt.xticks([-1, -0.5, 0, 0.5, 1], fontsize=15)
            plt.yticks([0, 0.4, 0.8], fontsize=15)
        elif mode == "white noise":
            results = its_class.noise_sort_stab(dt_euler, dt_intra, T, sigma_time)
            ffrec = results[0]
            its = results[1]
            plt.scatter(ffrec, its, c=colors[i], alpha=0.5, label="rank="+str(D))

    plt.legend()
    plt.savefig("F:/Downloads/fig.pdf", bbox_inches='tight')
    plt.show()



# Dimensionality
##############################################################################################################
def compare_dim_only_combi_inter(n_neuron, R, kappa, beta_dim, num_sample, mode):
    """
    Compare the dimensionality with J = a*J_sym + (1-a)*J_asym of varias a value.
    """
    a_values = np.linspace(0, 1, 5)
    colors = [cm.to_hex(plt.cm.tab20b(i)) for i in range(20)]
    # Here is the bijective projection from L to ffrec (0,1).
    plt.figure(figsize=(6,5))
    for i in range(len(a_values)):
        a = a_values[i]
        network = AN.CombiAsymNet(n_neuron, a, R)
        ffrec = np.linspace(0,1, int(n_neuron/2))
        dim_class = AO.Dimensionality(n_neuron, R, network)
        if mode == "real part":
            analytical_dim = dim_class.real_dim_to_ffrec(kappa, beta_dim)
            empir_dim = dim_class.real_dim_to_ffrec_empir(kappa, beta_dim, num_sample)
            # Plot the analytical dimensionality as line and the empirical dimensionality
            # as dots in the same color.
            plt.plot(ffrec, np.flip(analytical_dim), c=colors[i], alpha=0.5, label="a="+str(a))
            plt.scatter(ffrec, np.flip(empir_dim), c=colors[i], alpha=0.5, label="a="+str(a))
            plt.xticks([0, 0.5, 1], fontsize=15)
            plt.yticks([5, 10], fontsize=15)
        elif mode == "symmetrized":
            analytical_dim = dim_class.sym_dim_analytical(kappa, beta_dim)
            empir_dim = dim_class.sym_dim_empir(kappa, beta_dim, num_sample)
            plt.plot(ffrec, np.flip(analytical_dim), c=colors[i], alpha=0.6, label="a="+str(a))
            plt.scatter(ffrec, np.flip(empir_dim), c=colors[i], alpha=0.6, label="a="+str(a))
            plt.xticks([0, 0.5, 1], fontsize=15)
            plt.yticks([5, 10], fontsize=15)
        elif mode == "white noise":
            # Still apply J_sym for analytical dimensionality.
            #analytical_dim = dim_class.sym_dim_analytical(kappa, beta_dim)
            empir_dim = dim_class.noise_dim_empir(kappa, beta_dim, num_sample)
            #plt.plot(ffrec, np.flip(analytical_dim), c=colors[i], alpha=0.6, label="a="+str(a))
            plt.scatter(ffrec, np.flip(empir_dim), c=colors[i], alpha=0.6, label="a="+str(a))
            plt.xlabel("Feedforward recurrent alignment", fontsize=18)
            plt.ylabel("Dimensionality", fontsize=18)
            plt.xticks([0, 0.5, 1], fontsize=15)
            plt.yticks([4, 7, 10], fontsize=15)

    plt.legend(bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left", mode="expand", ncol=5 ,  fontsize=15)
    plt.savefig("F:/Downloads/fig_dim.pdf", bbox_inches='tight')
    plt.show()


def compare_dim_low_rank(n_neuron, R, kappa, beta_dim, num_sample, mode, sigma_1=1, sigma_2=1):
    # Variation on the rank of the interaction.
    rank_values = [1, 5]
    colors = [cm.to_hex(plt.cm.tab10(i)) for i in range(10)]
    # Here is the bijective projection from L to ffrec (0,1).
    plt.figure(figsize=(6,5))
    for i in range(len(rank_values)):
        D = rank_values[i]
        # Different network setup:
        #network = AN.LowRank(n_neuron, D, R, sigma_1, sigma_2)
        network = AN.NoisedLowRank(n_neuron, D, R)
        #network = SN.NoisedLowRank_1D(n_neuron, R)
        ffrec = np.linspace(0,1, int(n_neuron/2))
        dim_class = AO.Dimensionality(n_neuron, R, network)
        if mode == "real part":
            analytical_dim = dim_class.real_dim_to_ffrec(kappa, beta_dim)
            empir_dim = dim_class.real_dim_to_ffrec_empir(kappa, beta_dim, num_sample)
            # Plot the analytical dimensionality as line and the empirical dimensionality
            # as dots in the same color.
            plt.plot(ffrec, np.flip(analytical_dim), c=colors[i], alpha=0.5, label="rank="+str(D))
            plt.scatter(ffrec, np.flip(empir_dim), c=colors[i], alpha=0.5, label="rank="+str(D))
        elif mode == "symmetrized":
            analytical_dim = dim_class.sym_dim_analytical(kappa, beta_dim)
            empir_dim = dim_class.sym_dim_empir(kappa, beta_dim, num_sample)
            #ffrec_sym = dim_class.sym_ffrec()
            #plt.scatter(np.flip(ffrec_sym), np.flip(analytical_dim), c=colors[i], alpha=0.6, label="rank="+str(D))
            #plt.scatter(np.flip(ffrec_sym), np.flip(empir_dim), c="green", alpha=0.6, label="rank="+str(D))
            plt.plot(ffrec, np.flip(analytical_dim), c=colors[i], alpha=0.6, label="rank="+str(D))
            plt.scatter(ffrec, np.flip(empir_dim), c=colors[i], alpha=0.6, label="rank="+str(D))
            plt.xlabel("Feedfroward recurrent alignment", fontsize=18)
            plt.ylabel("Dimensionality", fontsize=18)
            plt.xticks([0, 0.5, 1], fontsize=15)
            plt.yticks([4, 6, 8, 10], fontsize=15)
        elif mode == "white noise":
            # Still apply J_sym for analytical dimensionality.
            analytical_dim = dim_class.sym_dim_analytical(kappa, beta_dim)
            empir_dim = dim_class.noise_dim_empir(kappa, beta_dim, num_sample)
            plt.plot(ffrec, np.flip(analytical_dim), c=colors[i], alpha=0.6, label="rank="+str(D))
            plt.scatter(ffrec, np.flip(empir_dim), c=colors[i], alpha=0.6, label="rank="+str(D))
    plt.legend()
    plt.savefig("F:/Downloads/fig.pdf", bbox_inches='tight')
    plt.show()

##############################################################################################################
# ALignment to spontaneous activity.
def compare_align_spont_combi(n_neuron, R, kappa, beta_spont, beta_dim, num_sample, mode):
    """
    Compare the alignment to spontaneous activity with J = a*J_sym + (1-a)*J_asym of varias a value.
    """
    a_values = np.linspace(0, 1, 5)
    colors = [cm.to_hex(plt.cm.tab20b(i)) for i in range(20)]
    plt.figure(figsize=(6, 5))
    for i in range(len(a_values)):
        a = a_values[i]
        network = AN.CombiAsymNet(n_neuron, a, R)
        align_class = AO.AlignmentSpontaneousAct(n_neuron, R, network)
        if mode == "real part":
            results = align_class.real_pattern_align(kappa, beta_spont, beta_dim, num_sample)
            ffrec = results[0]
            pattern_align = results[1]
            plt.scatter(ffrec, pattern_align, c=colors[i], alpha=0.5, label="a="+str(a))
        elif mode == "symmetrized":
            results = align_class.sym_pattern_align(kappa, beta_spont, beta_dim, num_sample)
            ffrec = results[0]
            pattern_align = results[1]
            plt.scatter(ffrec, pattern_align, c=colors[i], alpha=0.5, label="a="+str(a))
            plt.xlabel("Feedforward recurrent alignment", fontsize = 17)
            plt.ylabel("Alignment to spontaneous activity", fontsize=17)
            plt.xticks([0, 0.5, 1], fontsize=15)
            plt.yticks([0, 0.1, 0.2], fontsize=15)
        elif mode == "white noise":
            results = align_class.noise_pattern_align(kappa, beta_spont, beta_dim, num_sample)
            ffrec = results[0]
            pattern_align = results[1]
            plt.scatter(ffrec, pattern_align, c=colors[i], alpha=0.6, label="a="+str(a))
            plt.xlabel("Feedforward recurrent alignment", fontsize=18)
            plt.ylabel("Alignment to spont.act", fontsize=18)
            plt.xticks([0, 0.5, 1], fontsize=15)
            plt.yticks([0, 0.08, 0.16], fontsize=15)

    plt.legend(fontsize=15)
    plt.savefig("F:/Downloads/fig_align.pdf", bbox_inches='tight')
    plt.show()


def compare_align_spont_low_rank(n_neuron, R, kappa, beta_spont, beta_dim, num_sample, mode,
                                 sigma_1=1, sigma_2=1):
    # Variation on the rank of the interaction.
    rank_values = [1, 5]
    colors = [cm.to_hex(plt.cm.tab10(i)) for i in range(10)]
    plt.figure(figsize=(6,5))
    for i in range(len(rank_values)):
        D = rank_values[i]
        # Different network setup:
        #network = AN.LowRank(n_neuron, D, R, sigma_1, sigma_2)
        #network = SN.NoisedLowRank_1D(n_neuron, R)
        network = AN.NoisedLowRank(n_neuron, D, R)
        align_class = AO.AlignmentSpontaneousAct(n_neuron, R, network)
        if mode == "real part":
            results = align_class.real_pattern_align(kappa, beta_spont, beta_dim, num_sample)
            ffrec = results[0]
            pattern_align = results[1]
            plt.scatter(ffrec, pattern_align, c=colors[i], alpha=0.5, label="rank="+str(D))
        elif mode == "symmetrized":
            results = align_class.sym_pattern_align(kappa, beta_spont, beta_dim, num_sample)
            ffrec = results[0]
            pattern_align = results[1]
            plt.scatter(ffrec, pattern_align, c=colors[i], alpha=0.5, label="rank="+str(D))
            plt.xlabel("Feedforward recurrent alignment", fontsize=18)
            plt.ylabel("Alignment to spont. act", fontsize=18)
            plt.xticks([0, 0.5, 1], fontsize=15)
            plt.yticks([0.05, 0.1, 0.15, 0.2], fontsize=15)
        elif mode == "white noise":
            results = align_class.noise_pattern_align(kappa, beta_spont, beta_dim, num_sample)
            ffrec = results[0]
            pattern_align = results[1]
            plt.scatter(ffrec, pattern_align, c=colors[i], alpha=0.6, label="rank="+str(D))
    plt.legend()
    plt.savefig("F:/Downloads/fig.pdf", bbox_inches='tight')
    plt.show()

##############################################################################################################
if __name__ == "__main__":
    # parameters
    n_neuron = 200
    R = 0.85
    sigma_trial = 0.05 # determines the curve. 0.02 will lead to a "diagonal line".
    sigma_time = 0.3
    N_trial = 100
    dt_euler = 0.1
    dt_intra = 20
    T = 120
    kappa = 5
    beta_dim = 10
    beta_spont = 20
    num_sample = 500
    # For the case of low rank interaction.
    sigma_1 = 1
    sigma_2 = 2

# Commands, chnage mode for different setup.
##############################################################################################################
    # Results of applying combined interaction matrix J = a*J_sym + (1-a)*J_asym

    #compare_ttc_only_combi_inter(n_neuron, R, sigma_trial, N_trial, "white noise")

    #compare_its_only_combi_inter(n_neuron, R, dt_euler, dt_intra, T, sigma_time, "white noise")

    #compare_dim_only_combi_inter(n_neuron, R, kappa, beta_dim, num_sample, "real part")

    #compare_align_spont_combi(n_neuron, R, kappa, beta_spont, beta_dim, num_sample, "white noise")

##############################################################################################################

    # Results of applying row ranked interation matrix.

    #compare_ttc_low_rank(n_neuron, R, sigma_trial, N_trial, "symmetrized")

    #compare_its_low_rank(n_neuron, R, dt_euler, dt_intra, T, sigma_time, "symmetrized")

    #compare_dim_low_rank(n_neuron, R, kappa, beta_dim, num_sample, "symmetrized")

    #compare_align_spont_low_rank(n_neuron, R, kappa, beta_spont, beta_dim, num_sample, "symmetrized")






