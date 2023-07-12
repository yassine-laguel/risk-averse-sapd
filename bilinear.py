import numpy as np
from matplotlib import pyplot as plt
import os, sys
import json


def prox_f(x, tau, mux):
    return 1./(1+tau*mux) * x

def prox_g(y, sigma, muy):
    return 1./(1+sigma*muy) * y

def grad_x(x, y):
        return K @ y

def grad_y(x, y):
    return K @ x

def sto_grad_x(x, y):
    return grad_x(x,y) + np.random.multivariate_normal(mean, cov)

def sto_grad_y(x, y):
    return grad_y(x,y) + np.random.multivariate_normal(mean, cov)


def run_sapd(n_iter, theta, mux, muy):

    # Chambolle Pock parameterization
    tau = (1-theta)/(theta*mux)
    sigma = (1-theta)/(theta*muy)

    # Sequence of iterates
    iterates_x = np.zeros((n_iter, d))
    iterates_y = np.zeros((n_iter, d))

    # arbitrary starting point
    u, v = np.random.randn(d), np.random.randn(d)
    iterates_x[0] = 50 * u
    iterates_y[0] = 50 * v

    # Start run SAPD
    pgy = sto_grad_y(iterates_x[0], iterates_y[0])
    for ii in range(n_iter-1):
        gy = sto_grad_y(iterates_x[ii], iterates_y[ii])
        q = gy - pgy
        s = gy + theta * q
        iterates_y[ii+1] = prox_g(iterates_y[ii] + sigma * s, sigma, muy)
        iterates_x[ii+1] = prox_f(iterates_x[ii] - tau * sto_grad_x(iterates_x[ii], iterates_y[ii+1]), tau, mux)
        pgy = gy
    return iterates_x, iterates_y


def article_plot(array1, array2, array3, array4, array5, array6, bins=50,
                 title1=None, title2=None, title3=None,
                 xlabel1=None, xlabel2=None, xlabel3=None, ylabel=None,
                 title=None, labels=None):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Norms with fill_between
    avg1, max1, min1 = array1.mean(axis=0), array1.max(axis=0), array1.min(axis=0)
    avg2, max2, min2 = array2.mean(axis=0), array2.max(axis=0), array2.min(axis=0)

    axes[0].plot(avg1, label=labels[0] if labels else None, color='royalblue', linewidth=2)
    axes[0].plot(avg2, label=labels[1] if labels else None, color='darkorange', linewidth=2)

    axes[0].fill_between(range(len(max1)), min1, max1, color='royalblue', alpha=0.2)
    axes[0].fill_between(range(len(max2)), min2, max2, color='darkorange', alpha=0.2)
    axes[0].legend(loc='upper right', bbox_to_anchor=(1, 1))

    # Histograms
    hist_style1 = dict(alpha=0.3, color='royalblue',  edgecolor='black', linewidth=1.2)
    hist_style2 = dict(alpha=0.3, color='darkorange', edgecolor='black', linewidth=1.2)

    weight3 = np.ones_like(array3) / (len(array3))
    weight4 = np.ones_like(array4) / (len(array4))

    min_value3 = np.min([array3.min(), array4.min()])
    min_value4 = np.min([array5.min(), array6.min()])
    max_value3 = np.max([array3.max(), array4.max()])
    max_value4 = np.max([array5.max(), array6.max()])

    log_bins3 = np.logspace(np.log10(min_value3), np.log10(max_value3), bins)
    log_bins4 = np.logspace(np.log10(min_value4), np.log10(max_value4), bins)

    range3 = (min(array3.min(), array4.min()), max(array3.max(), array4.max()))
    range4 = (min(array5.min(), array6.min()), max(array5.max(), array6.max()))

    axes[1].hist(array3, bins=log_bins3, range=range3, weights=weight3, density=True, **hist_style1, label=labels[0] if labels else None)
    axes[1].hist(array4, bins=log_bins3, range=range3, weights=weight4, density=True, **hist_style2, label=labels[1] if labels else None)
    axes[2].hist(array5, bins=log_bins4, range=range4, weights=weight3, density=True, **hist_style1, label=labels[0] if labels else None)
    axes[2].hist(array6, bins=log_bins4, range=range4, weights=weight4, density=True, **hist_style2, label=labels[1] if labels else None)

    # Add vertical lines for the mean and 90th percentile
    mean3, mean4 = np.mean(array3), np.mean(array4)
    percentile90_3, percentile90_4 = np.percentile(array3, 90), np.percentile(array4, 90)
    axes[1].axvline(mean3, color='royalblue', linewidth=2)
    axes[1].axvline(mean4, color='darkorange', linewidth=2)
    axes[1].axvline(percentile90_3, color='royalblue', linestyle='--', linewidth=2)
    axes[1].axvline(percentile90_4, color='darkorange', linestyle='--', linewidth=2)

    mean5, mean6 = np.mean(array5), np.mean(array6)
    percentile90_5, percentile90_6 = np.percentile(array5, 90), np.percentile(array6, 90)
    axes[2].axvline(mean5, color='royalblue', linewidth=2)
    axes[2].axvline(mean6, color='darkorange', linewidth=2)
    axes[2].axvline(percentile90_5, color='royalblue', linestyle='--', linewidth=2)
    axes[2].axvline(percentile90_6, color='darkorange', linestyle='--', linewidth=2)

    # Add extra lines with the desired style to the last plot for legend purposes
    axes[2].plot([], [], color='black', linewidth=2, label='Mean')
    axes[2].plot([], [], color='black', linestyle='--', linewidth=2, label='90th percentile')

        # Set scales, labels, and title
    axes[0].set_yscale('log')
    axes[1].set_xscale('log')
    axes[1].set_yscale('log')
    axes[2].set_xscale('log')

    if title1:
        axes[0].set_title(title1, fontsize=14)
        axes[1].set_title(title2, fontsize=14)
        axes[2].set_title(title3, fontsize=14)

    if xlabel1:
        axes[0].set_xlabel(xlabel1, fontsize=14)
        axes[1].set_xlabel(xlabel2, fontsize=14)
        axes[2].set_xlabel(xlabel3, fontsize=14)
    if ylabel:
        axes[0].set_ylabel(ylabel, fontsize=12)
    if title:
        fig.suptitle(title, fontsize=18)

    # Customize ticks
    for ax in axes:
        ax.tick_params(axis='both', labelsize=12)

    # Create a single legend for all plots
#     if labels:
#         lines, labels = fig.axes[-1].get_legend_handles_labels()
#         legend = fig.legend(lines, labels, loc='lower center', ncol=len(labels), bbox_to_anchor=(0.5, -0.15), fontsize=12)

    # Adjust position of xlabel for first plot
    axes[0].xaxis.set_label_coords(0.5, -0.2)

#     if labels:
#         lines, labels = fig.axes[-1].get_legend_handles_labels()
#         legend = fig.legend(lines, labels, loc='lower center', ncol=len(labels), bbox_to_anchor=(0.5, -0.1), fontsize=12)

    # Adjust the layout and display the plot
    plt.tight_layout()
    if not os.path.exists('figs'):
        os.makedirs('figs')
    fig.savefig('figs/bilinear_game_perfs.pdf', bbox_inches='tight')
    plt.subplots_adjust(top=0.9, bottom=0.3)
    plt.show()


if __name__ == '__main__':
    # Fix seed
    np.random.seed(1)

    # Dimension of the problem
    d = 30

    # Generate a random matrix K
    K = np.random.rand(d, d)

    # Make K symmetric
    K = 10/(np.linalg.norm(K)) * K

    # Noise parameters
    delta = 1.
    mean = np.zeros(d)
    cov = delta**2 / d * np.eye(d)

    # Regularization parameters
    mux, muy = 1., 1.

    # Spectral radius of K
    spectral_radius = np.max(np.abs( np.linalg.eigvals(K)))
    kappa_max = spectral_radius / np.sqrt(mux*muy)

    # Lower bound on theta for convergence
    bar_theta = (np.sqrt(1 + kappa_max**2) - 1) / kappa_max
    theta = (1+ bar_theta)/2

    # iterations
    intermediate_it = 2000
    n_samples = 500
    n_iter = 5000
    theta_vals = [bar_theta, 1 - (1-bar_theta)**2]

    arr1 = np.zeros((n_samples, n_iter))
    arr2 = np.zeros((n_samples, n_iter))

    for jj in range(n_samples):
        sys.stdout.write('%d samples completed \r' % jj)
        sys.stdout.flush()
        iterates_x1, iterates_y1 = run_sapd(n_iter, theta_vals[0], mux, muy)
        iterates_x2, iterates_y2 = run_sapd(n_iter, theta_vals[1], mux, muy)
        iterates_norm1 = [np.linalg.norm(u)**2 + np.linalg.norm(v)**2 for (u,v) in zip(iterates_x1, iterates_y1)]
        iterates_norm2 = [np.linalg.norm(u)**2 + np.linalg.norm(v)**2 for (u,v) in zip(iterates_x2, iterates_y2)]
        arr1[jj] = np.array(iterates_norm1)
        arr2[jj] = np.array(iterates_norm2)

    performances_dct = {}
    performances_dct['rho1'] = arr1.tolist()
    performances_dct['rho2'] = arr2.tolist()

    # write optimal solution to JSON file
    if not os.path.exists('data'):
        os.makedirs('data')
    with open('data/bilinear_performances.json', 'w') as f:
        json.dump(performances_dct, f)

    with open('data/bilinear_performances.json', 'r') as f:
         performances_dct = json.load(f)

    title1 = r'Bilinear Game - convergence'
    title2 = r'Bilinear Game - intermediate histogram'
    title3 = r'Bilinear Game - final histogram'

    arr1 = np.array(performances_dct['rho1'])
    arr2 = np.array(performances_dct['rho2'])
    arr3 = arr1[:, intermediate_it]
    arr4 = arr2[:, intermediate_it]
    arr5 = arr1[:, -1]
    arr6 = arr2[:, -1]

    article_plot(arr1, arr2, arr3, arr4, arr5, arr6, bins=50,
                 title1=title1, title2=title2, title3=title3,
                 xlabel1='iterations',
                 xlabel2=r"$\Vert y_k - x^\star \Vert^2 + \Vert y - y^\star\Vert^2$, $k=2000$",
                 xlabel3=r"$\Vert y_k - x^\star \Vert^2 + \Vert y - y^\star\Vert^2$, $k=5000$",
                 ylabel=r'$\Vert x_k - x^\star \Vert^2+ \Vert y_k - y^\star\Vert^2$', title=None,
                 labels=[r'$\rho_1=.892$', r'$\rho_2=.988$'])
