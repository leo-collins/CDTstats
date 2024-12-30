import matplotlib.pyplot as plt
import numpy as np
from numpy.random import f

def box_plot_particles(log_likelihoods_pf: dict, figname = None):
    plt.figure(figsize=(12, 8))

    labels = []
    data = []
    for (num_particles, num_runs), likelihoods in log_likelihoods_pf.items():
        labels.append(f"Particles: {num_particles}, Runs: {num_runs}")
        data.append(likelihoods)
    plt.boxplot(data, vert=False, patch_artist=True, labels=labels)
    plt.xlabel("Log-marginal Likelihood")
    plt.title("Box Plot of Log-marginal Likelihood Estimates with Different Particle Settings")
    plt.grid()
    if figname:
        plt.savefig(f"report/figs/box_plot_{figname}.jpg")
    plt.show()

def plot_particle_convergence(log_likelihoods_pf, log_likelihood_kf, figname = None):
    num_particles_list = [key[0] for key in log_likelihoods_pf.keys()]
    mean_log_likelihoods = [np.mean(log_likelihoods_pf[key]) for key in log_likelihoods_pf.keys()]

    plt.figure(figsize=(10, 6))
    plt.scatter(num_particles_list, mean_log_likelihoods, label="Particle Filter Log Likelihoods", color="blue")
    plt.axhline(y=log_likelihood_kf, color='red', linestyle='--', label="Kalman Filter Log Likelihood")

    plt.xscale('log')
    plt.xlabel("Number of Particles")
    plt.ylabel("Log Likelihood")
    plt.title("Convergence of Particle Filter Log Likelihoods to Kalman Filter Log Likelihood")
    plt.legend()
    plt.grid(True)
    if figname:
        plt.savefig(f"report/figs/convergence_{figname}.jpg")
    plt.show()

def plot_particle_loglog(log_likelihoods_pf, log_likelihood_kf, particles = True, mse=True, figname = None):
    if particles:
        num = [key[0] for key in log_likelihoods_pf.keys()]
        plot_type = "particles"
    else:
        num = [key[1] for key in log_likelihoods_pf.keys()]
        plot_type = "runs"
    if mse:
        errors = [np.sum((run - log_likelihood_kf)**2) / len(run) for run in log_likelihoods_pf.values()]
        error_type = "MSE"
    else:
        errors = [np.linalg.norm(run - log_likelihood_kf) for run in log_likelihoods_pf.values()]
        error_type = "L2 error"
    plt.figure(figsize=(10, 6))
    plt.loglog(num, errors, marker='o', label="mean squared error of Particle Filter Likelihoods")

    slope, intercept = np.polyfit(np.log(num), np.log(errors), 1)

    plt.loglog(num, np.exp(intercept) * np.array(num)**slope, linestyle='--', label=f"Fitted Line (slope = {slope:.2f})")

    plt.xlabel(f"Number of {plot_type}")
    plt.ylabel(f"{error_type}")
    plt.legend()
    plt.grid(True)
    if figname:
        plt.savefig(f"report/figs/loglog_{figname}.jpg")
    plt.show()

    print(f"Convergence rate (slope of the line): {slope:.2f}")

def plot_runs_loglog(log_likelihoods_pf, log_likelihood_kf, mse=True):
    num_runs = [key[1] for key in log_likelihoods_pf.keys()]

    if mse:
        errors = [np.sum((run - log_likelihood_kf)**2) / len(run) for run in log_likelihoods_pf.values()]
    else:
        errors = [np.linalg.norm(run - log_likelihood_kf) for run in log_likelihoods_pf.values()]
    

    plt.figure(figsize=(10, 6))
    plt.loglog(num_runs, errors, marker='o', label="L2 error")

    slope, icpt = np.polyfit(np.log(num_runs), np.log(errors), 1)

    plt.loglog(num_runs, np.exp(icpt) * np.array(num_runs)**slope, linestyle='--', label=f"slope = {slope:.2f}")

    plt.xlabel("Number of runs")
    plt.ylabel("L2 Error")
    plt.legend()
    plt.grid(True)
    plt.show()