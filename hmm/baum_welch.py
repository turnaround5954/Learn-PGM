import numpy as np
from scipy.stats import multivariate_normal


def e_step(X, pi, A, mu, sigma2):
    """E-step: forward-backward message passing"""
    # Messages and sufficient statistics
    N, T, K = X.shape
    M = A.shape[0]
    alpha = np.zeros([N, T, M])  # [N,T,M]
    alpha_sum = np.zeros([N, T])  # [N,T], normalizer for alpha
    beta = np.zeros([N, T, M])  # [N,T,M]
    gamma = np.zeros([N, T, M])  # [N,T,M]
    xi = np.zeros([N, T-1, M, M])  # [N,T-1,M,M]

    # Forward messages
    diff = X[:, :, None, :] - mu  # [N,T,M,K]
    covs = np.kron(sigma2[:, None, None], np.eye(K))  # [M,K,K]
    prcs = np.linalg.inv(covs)
    exponents = np.sum(np.einsum('ntmi,mij->ntmj', diff, prcs) * diff, axis=-1)
    emissions = np.exp(-0.5 * exponents) / \
        np.sqrt(np.linalg.det(covs) * (2 * np.pi) ** K)  # [N,T,M]

    alpha[:, 0, :] = pi * emissions[:, 0, :]
    alpha_sum[:, 0] = alpha[:, 0, :].sum(axis=-1)
    alpha[:, 0, :] /= alpha_sum[:, 0, None]
    for t in range(T - 1):
        alpha[:, t+1, :] = alpha[:, t, :] @ A * emissions[:, t+1, :]
        alpha_sum[:, t+1] = alpha[:, t+1, :].sum(axis=-1)
        alpha[:, t+1, :] /= alpha_sum[:, t+1, None]

    # Backward messages
    beta[:, T-1, :] = 1
    for t in range(T - 1, 0, -1):
        beta[:, t-1, :] = (beta[:, t, :] * emissions[:, t, :]) @ A.T
        beta[:, t-1, :] /= alpha_sum[:, t, None]

    # Sufficient statistics
    gamma = alpha * beta
    xi = alpha[:, :-1, :, None] * (beta * emissions)[:, 1:, None, :] * A
    xi /= alpha_sum[:, 1:, None, None]  # implicit normalization

    # Although some of them will not be used in the M-step, please still
    # return everything as they will be used in test cases
    return alpha, alpha_sum, beta, gamma, xi


def m_step(X, gamma, xi):
    """M-step: MLE"""
    # TODO ...
    pi = gamma[:, 0, :].mean(axis=0)
    pi = pi / pi.sum(axis=-1)
    A = xi.sum(axis=(0, 1)) / (gamma[:, :-1, :].sum(axis=(0, 1))[:, None])

    N, T, K = X.shape
    sum_gamma = gamma.sum(axis=(0, 1))
    mu = np.einsum('ntm,ntk->mk', gamma, X) / sum_gamma[:, None]
    diff = X[:, :, None, :] - mu
    sigma2 = np.einsum('ntmk,ntmk->m', diff, diff *
                       gamma[:, :, :, None]) / sum_gamma / K

    return pi, A, mu, sigma2


def hmm_train(X, pi, A, mu, sigma2, em_step=20):
    """Run Baum-Welch algorithm."""
    for step in range(em_step):
        _, alpha_sum, _, gamma, xi = e_step(X, pi, A, mu, sigma2)
        pi, A, mu, sigma2 = m_step(X, gamma, xi)
        print(f"step: {step}  ln p(x): {np.einsum('nt->', np.log(alpha_sum))}")
    return pi, A, mu, sigma2


def hmm_generate_samples(N, T, pi, A, mu, sigma2):
    """Given pi, A, mu, sigma2, generate [N,T,K] samples."""
    M, K = mu.shape
    Y = np.zeros([N, T], dtype=int)
    X = np.zeros([N, T, K], dtype=float)
    for n in range(N):
        Y[n, 0] = np.random.choice(M, p=pi)  # [1,]
        X[n, 0, :] = multivariate_normal.rvs(
            mu[Y[n, 0], :], sigma2[Y[n, 0]] * np.eye(K))  # [K,]
    for t in range(T - 1):
        for n in range(N):
            Y[n, t+1] = np.random.choice(M, p=A[Y[n, t], :])  # [1,]
            # [K,]
            X[n, t+1, :] = multivariate_normal.rvs(
                mu[Y[n, t+1], :], sigma2[Y[n, t+1]] * np.eye(K))
    return X


def main():
    """Run Baum-Welch on a simulated toy problem."""
    # Generate a toy problem
    np.random.seed(12345)  # for reproducibility
    N, T, M, K = 10, 100, 4, 2
    pi = np.array([.0, .0, .0, 1.])  # [M,]
    A = np.array([[.7, .1, .1, .1],
                  [.1, .7, .1, .1],
                  [.1, .1, .7, .1],
                  [.1, .1, .1, .7]])  # [M,M]
    mu = np.array([[2., 2.],
                   [-2., 2.],
                   [-2., -2.],
                   [2., -2.]])  # [M,K]
    sigma2 = np.array([.2, .4, .6, .8])  # [M,]
    X = hmm_generate_samples(N, T, pi, A, mu, sigma2)

    # Run on the toy problem
    pi_init = np.random.rand(M)
    pi_init = pi_init / pi_init.sum()
    A_init = np.random.rand(M, M)
    A_init = A_init / A_init.sum(axis=-1, keepdims=True)
    mu_init = 2 * np.random.rand(M, K) - 1
    sigma2_init = np.ones(M)
    pi, A, mu, sigma2 = hmm_train(
        X, pi_init, A_init, mu_init, sigma2_init, em_step=20)
    print(pi)
    print(A)
    print(mu)
    print(sigma2)


if __name__ == '__main__':
    main()
