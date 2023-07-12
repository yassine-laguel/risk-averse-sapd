import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from numba import njit
from numba.core import types
from numba.typed import Dict

import os, sys, json, math


# Fetch dataset
path_file = "data/dry_bean.csv"
dry_bean_data = pd.read_csv(path_file, header=None)

# Separate input-output and convert into numpy array
input_data = dry_bean_data.iloc[:, :-1]  # All columns except the last one
output_data = dry_bean_data.iloc[:, -1]  # Only the last column

A = input_data.to_numpy()

n, d = A.shape
for i in range(d):
    A[:, i] = (A[:, i] - np.min(A[:, i])) / (np.max(A[:, i]) - np.min(A[:, i]))

b_non_encoded = output_data.to_numpy()
label_encoder = LabelEncoder()
label_encoder.fit(output_data)
b = label_encoder.transform(output_data)
b[b==0] = -1

A_train, A_test, b_train, b_test = train_test_split(A, b, test_size=0.3, random_state=42)

# Update A and b with the training data
A = np.ascontiguousarray(A_train)
b = np.ascontiguousarray(b_train)
n, d = A.shape

@njit
def f_loss(x, y, A, b, lambda_, mu_y):
    # Compute the objective value
    x = np.ascontiguousarray(x)
    y = np.ascontiguousarray(y)
    mid_value = -(A @ x) * b
    is_inf = False
    for val in mid_value.flatten():
        exp_val = math.exp(val)
        if exp_val > 1e300:  # A large threshold value to check for infinity
            is_inf = True
            break
    if is_inf:
        res = y.T @ mid_value + lambda_ / 2 * x.T @ x - mu_y / 2 * y.T @ y
    else:
        res = y.T @ np.log(1 + np.exp(mid_value)) + lambda_ / 2 * x.T @ x - mu_y / 2 * y.T @ y
    return res

@njit
def l_val(i, x, A, b):
    x = np.ascontiguousarray(x)
    mid_value = -b[i] * A[i, :] @ x
    if np.isinf(np.exp(mid_value)):
        value = mid_value
    else:
        value = np.log(1 + np.exp(mid_value))
    return value

@njit
def l_grad(i, x, A, b):
    x = np.ascontiguousarray(x)
    mid_value = -b[i] * A[i, :] @ x
    if np.isinf(np.exp(mid_value)):
        value = -b[i] * A[i, :]
    else:
        value = -b[i] * np.exp(mid_value) / (1 + np.exp(mid_value)) * A[i, :]
    return value

@njit
def predict_logistic(x, A, b):
    # Predict function given training result x and data matrix A
    # label b is to compute error
    n = A.shape[0]
    predict = np.zeros(n)
    x = np.ascontiguousarray(x)
    for i in range(n):
        val = A[i] @ x
        if val >= 0:
            predict[i] = 1
        else:
            predict[i] = -1
    error = 1 - np.sum(b == predict) / len(b)
    return error

@njit
def proj_g(p, lambda_, rho):
    p = (1 / (lambda_ + 1)) * p
    u = np.sort(p)[::-1]
    n = len(u)
    r = 2 * rho / n**2

    b = 0
    for k in range(1, n + 1):
        b += u[k - 1]
        if (b - 1) / k < u[k - 1]:
            K = k
        else:
            break

    tau = (b - u[k - 1] - 1) / K
    res = np.maximum(p - tau, 0)

    if np.linalg.norm(res)**2 <= r + 1 / n:
        return res
    else:
        u = np.append(u, [0])
        a = u[0]**2
        b = u[0]
        for k in range(2, n + 1):
#         for k in range(1, n):
            a += u[k - 1] * u[k - 1]
            b += u[k - 1]
#             if (a - b**2 / k) != 0 and k == 1:
            c = (r + 1 / n - 1 / k) / (a - b**2 / k)
#             if c >= 0 and (b - k * u[k]) != 0 and (b - k * u[k + 1])!= 0:
            if c >= 0:
                lambda_ = np.sqrt(c)
#                 upper_bound = 1 / (b - k * u[k])
                upper_bound = 1 / (b - k * u[k-1])
                if k == n:
                    lower_bound = 0
                else:
#                     lower_bound = 1 / (b - k * u[k + 1])
                    lower_bound = 1 / (b - k * u[k])

                if lower_bound <= lambda_ <= upper_bound:
                    res = np.maximum(0, lambda_ * p - ((b * lambda_) - 1) / k)
                    return res
    return res

@njit
def proj_f(x, lambda_, R):
    p = (1 / (lambda_ + 1)) * x
    norm_of_p = np.linalg.norm(p)

    if norm_of_p <= R:
        res = p
    else:
        res = p / norm_of_p * R

    return res

def SAPD(para, seed=1, log_bar=False):
    np.random.seed(seed)
    # data initial
    b = para['b']
    A = para['A']
    n, d = A.shape
    K = para['maxIter']
    zero_vector = para['zero_vector']
    m_x = para['batchsize_x']
    m_y = para['batchsize_y']
    rho = para['rho']  # convergence rate
    Rho = para['Rho']  # ||y||^2<= 2*Rho/n^2
    R = para['R']  # diameter of x
    lambda_ = para['lambda']  # regulizer for x
    mu_y = para['mu_y']  # regulizer for y

    # stepsize
    theta = para['theta']
    tau = para['tau']
    sigma = para['sigma']

    # data initial
    x = para['x']
    y = para['y']
    x[:, 0] = para['x0']
    y[:, 0] = para['y0']
    x[:, 1] = para['x0']
    y[:, 1] = para['y0']
    if log_bar:
        x_bar = np.copy(x)
        y_bar = np.copy(y)
        K_N = np.zeros(K + 1)
        K_N[1] = 0

    for k in range(1, K):
#         sys.stdout.write('%d iterations completed \r' % k)
        if k % 100 == 0:
            sys.stdout.write('seed {s} : {it} iterations completed \r'.format(s=seed, it=k))
            sys.stdout.flush()
        # update for y
        if k == 1:
            Gradient_y_1 = zero_vector
            batch_1 = np.random.permutation(n)[:m_y]
            for i in batch_1:
                Gradient_y_1[i] = l_val(i, x[:, k - 1], A, b)
        else:
            Gradient_y_1 = Gradient_y_2

        batch_2 = np.random.permutation(n)[:m_y]
        Gradient_y_2 = zero_vector
        for i in batch_2:
            Gradient_y_2[i] = l_val(i, x[:, k], A, b)

        P_center = y[:, k] + sigma * (1 + theta) * (Gradient_y_2 * n / m_y) - sigma * theta * (Gradient_y_1 * n / m_y)
        y[:, k + 1] = proj_g(P_center, mu_y * sigma, Rho)

        # update for x
        batch_3 = np.random.permutation(n)[:m_x]
        Gradient_x = 0
        for i in batch_3:
            Gradient_x = Gradient_x + y[i, k + 1] * l_grad(i, x[:, k], A, b)

        P_center = x[:, k] - tau * (Gradient_x * n / m_x)
        x[:, k + 1] = proj_f(P_center, lambda_ * tau, R)
        if log_bar:
            K_N[k + 1] = K_N[k] + rho ** (-(k - 1))
            x_bar[:, k + 1] = (x_bar[:, k] * K_N[k] + x[:, k + 1] * rho ** (-(k - 1))) / K_N[k + 1]
            y_bar[:, k + 1] = (y_bar[:, k] * K_N[k] + y[:, k + 1] * rho ** (-(k - 1))) / K_N[k + 1]

    result = {'x': x, 'y': y}
    if log_bar:
        result['x_bar'] = x_bar
        result['y_bar'] = y_bar

    error = np.zeros(int(K/5))
    lossvalue = np.zeros(int(K/5))
    if log_bar:
        error_bar = np.zeros(int(K/5))
        lossvalue_bar = np.zeros(int(K/5))

    for k in range(K):
        if k % 5 == 0:
            if k % 1000 == 0:
                fixed_length = 30
                sys.stdout.write('seed {s} - metrics : {it} iterations completed \r'.format(s=seed, it=k))
                sys.stdout.flush()
            error[int(k/5)] = predict_logistic(x[:, k], A, b)
            lossvalue[int(k/5)] = f_loss(x[:, k], y[:, k], A, b, lambda_, mu_y)
            if log_bar:
                error_bar[int(k/5)] = predict_logistic(x_bar[:, k], A, b)
                lossvalue_bar[int(k/5)] = f_loss(x_bar[:, k], y_bar[:, k], A, b, lambda_, mu_y)

    result['error'] = error
    result['lossvalue'] = lossvalue
    if log_bar:
        result['error_bar'] = error_bar
        result['lossvalue_bar'] = lossvalue_bar

    return result


def gap_function(x, y, x_opt, y_opt):
    lambda_ = para['lambda']
    mu_y = para['mu_y']
    term1 = f_loss(x_opt, y, A, b, lambda_, mu_y)
    term2 = f_loss(x, y_opt, A, b, lambda_, mu_y)
    return term2 - term1

def gap_performances(iterates_x, iterates_y, x_opt, y_opt):
    gap_values = [gap_function(iterates_x[:, ii], iterates_y[:, ii], x_opt, y_opt) for ii in range(iterates_x.shape[1])]
    return gap_values

def distance_performances(iterates_x, iterates_y, x_opt, y_opt):
    dist_values = [np.linalg.norm(iterates_x[:, ii]-x_opt)**2 + np.linalg.norm(iterates_y[:, ii]-y_opt)**2 for ii in range(iterates_x.shape[1])]
    return dist_values

# Misclassification error
def misc_error(model, A, b):
    predictions = ((A @ model) >= 0.).astype(float)
    predictions[predictions==0] = -1
    return 1. - np.sum(predictions == b) / len(b)

def generate_param_dct(rho, max_iter=5000, batch_size_x=1, batch_size_y=1):
    # Set running time, batchsize, and simulation num
    para = {}
    para['maxIter'] = max_iter
#     para['maxIter'] = 10
    batchsize = [10]
    sim_num = 50

    # Set regulizer and strongly convex parameter
    para['lambda'] = 0.01
    para['mu_x'] = 0.01
    para['mu_y'] = 10

    # Set projection diameter
    para['Rho'] = np.sqrt(n)
    para['R'] = np.sqrt(d)

    # Set the variance of the noise
    para['delta_y'] = 10
    para['delta_x'] = 10
    para['delta'] = 10

    # Find lipshitz constant and other parameters
    para['A'] = A
    para['b'] = b
    para['L_yx'] = np.linalg.norm(A)
    para['L_xy'] = np.linalg.norm(A)
    para['L_yy'] = 0
    para['L_xx'] = 1/4 * np.max(np.sqrt(np.sum(A**2, axis=1)))

    # Initialize iteration sequence
    para['x'] = np.ascontiguousarray(np.zeros((d, para['maxIter']+1)))
    para['y'] = np.ascontiguousarray(np.zeros((n, para['maxIter']+1)))
    para['x0'] = 2 * np.ones(d)
    para['y0'] = (1/n) * np.ones(n)
    para['zero_vector'] = np.zeros(n)

    # Algorrithm parameters
    para['rho'] = rho
    para['theta'] = rho
    para['tau'] = (1 - rho) / (para['mu_x'] * rho)
    para['sigma'] = (1 - rho) / (para['mu_y'] * rho)

    para['batchsize_x'] = batch_size_x
    para['batchsize_y'] = batch_size_y

    return para

def get_histogram_data(perf_dct, intermediary_iteration):

    pd_rho1 = performances_dct['rho1']
    rho_1_intermediary = np.array([pd_rho1[str(ii)][intermediary_iteration] for ii in range(n_samples)])
    rho_1_final = np.array([pd_rho1[str(ii)][-1] for ii in range(n_samples)])

    pd_rho2 = performances_dct['rho2']
    rho_2_intermediary = np.array([pd_rho2[str(ii)][intermediary_iteration] for ii in range(n_samples)])
    rho_2_final = np.array([pd_rho2[str(ii)][-1] for ii in range(n_samples)])

    return rho_1_intermediary, rho_2_intermediary, rho_1_final, rho_2_final

def article_plot(array1, array2, array3, array4, array5, array6, bins=50, title1=None, title2=None, title3=None, xlabel1=None, xlabel2=None, xlabel3=None, ylabel=None, title=None, labels=None):
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

    # Adjust position of xlabel for first plot
    axes[0].xaxis.set_label_coords(0.5, -0.2)

    #     if labels:
    #         lines, labels = fig.axes[-1].get_legend_handles_labels()
    #         legend = fig.legend(lines, labels, loc='lower center', ncol=len(labels), bbox_to_anchor=(0.5, -0.1), fontsize=12)

    # Adjust the layout and display the plot
    plt.tight_layout()
    if not os.path.exists('figs'):
        os.makedirs('figs')
    fig.savefig('figs/dry_bean_perfs.pdf', bbox_inches='tight')
    plt.subplots_adjust(top=0.9, bottom=0.3)
    plt.show()

if __name__ == '__main__':
    rho_opt = 0.98
    max_iter = 1000
    full_batch = A.shape[0]

    para = generate_param_dct(rho_opt, max_iter=max_iter,
                              batch_size_x=full_batch, batch_size_y=full_batch)
    result_opt = SAPD(para, log_bar=True)

    x_bar_opt = result_opt['x_bar'][:, -1]
    y_bar_opt = result_opt['y_bar'][:, -1]
    x_opt = result_opt['x'][:, -1]
    y_opt = result_opt['y'][:, -1]

    print('Misclassication error of x_opt : {}'.format(misc_error(x_opt, A, b)))
    print('Misclassication error of x_bar_opt : {}'.format(misc_error(x_bar_opt, A, b)))

    gap = gap_performances(result_opt['x'], result_opt['y'], x_bar_opt, y_bar_opt)
    gap_bar = gap_performances(result_opt['x_bar'], result_opt['y_bar'], x_bar_opt, y_bar_opt)

    # convert numpy arrays to lists
    x_bar_opt_lst = x_bar_opt.tolist()
    y_bar_opt_lst = y_bar_opt.tolist()

    # create a dictionary to store the lists
    data = {'x_bar_opt': x_bar_opt_lst, 'y_bar_opt': y_bar_opt_lst}

    # write optimal solution to JSON file
    with open('data/dry_bean_opt_solution.json', 'w') as f:
        json.dump(data, f)


    with open('data/dry_bean_opt_solution.json', 'r') as f:
        data = json.load(f)

        # convert lists to numpy arrays
    x_bar_opt = np.array(data['x_bar_opt'])
    y_bar_opt = np.array(data['y_bar_opt'])

    # Step 3.2.2

        # Parameters
    max_iter=4000
    intermediary_iteration = 500
    rho1 = 0.997
    n_samples = 500

        # Running
    performances_dct = {
        'rho1' : {},
        'rho2' : {},
    }
    para1 = generate_param_dct(rho1, max_iter=max_iter, batch_size_x=1, batch_size_y=1)

    for seed in range(n_samples):
        result1 = SAPD(para1, seed=seed)

        # Distance to optimal solutions:
        d1 = distance_performances(result1['x'], result1['y'], x_bar_opt, y_bar_opt)

        # Logging
        performances_dct['rho1'][str(seed)] = d1

    max_iter=4000
    rho2 = 0.999
    n_samples = 500

    para2 = generate_param_dct(rho2, max_iter=max_iter, batch_size_x=1, batch_size_y=1)

    for seed in range(n_samples):
        result2 = SAPD(para2, seed=seed)

        # Distance to optimal solutions:
        d2 = distance_performances(result2['x'], result2['y'], x_bar_opt, y_bar_opt)

        # Logging
        performances_dct['rho2'][str(seed)] = d2

    # write optimal solution to JSON file
    with open('data/dry_bean_performances.json', 'w') as f:
        json.dump(performances_dct, f)


    # Iterates race
    data_rho1 = np.array([performances_dct['rho1'][str(ii)] for ii in range(n_samples)])
    data_rho2 = np.array([performances_dct['rho2'][str(ii)] for ii in range(n_samples)])

    with open('data/dry_bean_performances.json', 'r') as f:
         performances_dct = json.load(f)

    title1 = r'Dry Bean - convergence'
    title2 = r'Dry Bean - intermediate histogram'
    title3 = r'Dry Bean - final histogram'


    arr1 = np.array([performances_dct['rho1'][str(ii)] for ii in range(n_samples)])
    arr2 = np.array([performances_dct['rho2'][str(ii)] for ii in range(n_samples)])
    arr3, arr4, arr5, arr6 = get_histogram_data(performances_dct, intermediary_iteration)
    article_plot(arr1, arr2, arr3, arr4, arr5, arr6, bins=50,
                 title1=title1, title2=title2, title3=title3,
                 xlabel1='iterations',
                 xlabel2=r"$\Vert y_k - x^\star \Vert^2 + \Vert y - y^\star\Vert^2$, $k=500$",
                 xlabel3=r"$\Vert y_k - x^\star \Vert^2 + \Vert y - y^\star\Vert^2$, $k=4000$",
                 ylabel=r'$\Vert x_k - x^\star \Vert^2+ \Vert y_k - y^\star\Vert^2$', title=None,
                 labels=[r'$\rho_1=.997$', r'$\rho_2=.999$'])
