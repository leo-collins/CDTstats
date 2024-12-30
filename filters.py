import numpy as np
from multiprocessing import Pool

def kalman_filter(y, A, Q, C, R, mu_0, Sigma_0):
    """
    Applies the Kalman filter to a sequence of observations.

    Parameters:
    y (np.ndarray): Observation sequence of shape (T, 2), where T is the number of time steps.
    A (np.ndarray): State transition matrix
    Q (np.ndarray): Process noise covariance matrix
    C (np.ndarray): Observation matrix
    R (np.ndarray): Observation noise covariance matrix
    mu_0 (np.ndarray): Initial state estimate
    Sigma_0 (np.ndarray): Initial state covariance matrix

    Returns:
    tuple: A tuple containing:
        - mu (np.ndarray): State estimates of shape (T, 2).
        - Sigma (np.ndarray): State covariance matrices of shape (T, 2, 2).
        - log_likelihood (float): Log-likelihood of the observations given the model.
    """
    T = y.shape[0]
    dim_x = A.shape[0]
    mu = np.zeros((T, dim_x))
    Sigma = np.zeros((T, dim_x, dim_x))
    log_likelihood = 0

    # Initial values
    mu_t = mu_0
    Sigma_t = Sigma_0

    for t in range(T):
        # Prediction step
        mu_pred = A @ mu_t
        Sigma_pred = A @ Sigma_t @ A.T + Q

        # Update step
        y_diff = y[t] - C @ mu_pred  # Innovation
        S = C @ Sigma_pred @ C.T + R  # Innovation covariance
        K = Sigma_pred @ C.T @ np.linalg.inv(S)  # Kalman gain
        mu_t = mu_pred + K @ y_diff  # State estimate
        Sigma_t = Sigma_pred - K @ C @ Sigma_pred  # State covariance

        # Compute log-likelihood
        log_likelihood += -0.5 * (
            np.log(np.linalg.det(S)) + y_diff.T @ np.linalg.inv(S) @ y_diff + 2 * np.log(2 * np.pi)
        )

        mu[t] = mu_t
        Sigma[t] = Sigma_t

    return mu, Sigma, log_likelihood

def particle_filter(y, A, Q, C, R, mu_0, Sigma_0, N):
    T = y.shape[0]
    particles = np.random.multivariate_normal(mu_0, Sigma_0, size=N)
    weights = np.ones(N) / N

    x_est = np.zeros((T, 2))
    log_likelihood = 0

    for t in range(T):
        # Evolve particles according to state transition model
        particles_pred = (A @ particles.T).T + np.random.multivariate_normal(np.zeros(2), Q, size=N)

        # Compute weights based on observation likelihood
        y_diff = y[t] - np.dot(particles_pred, C.T)
        likelihoods = np.exp(-0.5 * np.sum(y_diff @ np.linalg.inv(R) * y_diff, axis=1))
        likelihoods /= np.sqrt((2 * np.pi)**2 * np.linalg.det(R))

        weights *= likelihoods
        weights /= np.sum(weights)  # normalise weights

        log_likelihood += np.log(np.sum(likelihoods) / N)

        # Estimate state
        x_est[t] = np.average(particles_pred, axis=0, weights=weights)

        # Resample particles
        particles = particles_pred[np.random.choice(N, N, p=weights)]
        weights = np.ones(N) / N

    return x_est, log_likelihood

def particle_filter_runs(y, A, Q, C, R, mu_0, Sigma_0, num_particles, num_runs):
    """Runs particle filter multiple times
    """
    log_likelihoods = []
    with Pool() as p:
        log_likelihood_pf = p.starmap(particle_filter, [(y, A, Q, C, R, mu_0, Sigma_0, num_particles)] * num_runs)
    for run in log_likelihood_pf:
        log_likelihoods.append(run[1])
    return np.array(log_likelihoods)

