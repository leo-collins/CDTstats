import matplotlib.pyplot as plt
import numpy as np

def box_plot_particles(pf_results: dict, figname = None):
    plt.figure(figsize=(12, 8))

    labels = []
    data = []
    for (num_particles, num_runs), runs in pf_results.items():
        labels.append(f"Particles: {num_particles}, Runs: {num_runs}")
        data.append([run[1] for run in runs])
    plt.boxplot(data, vert=False, patch_artist=True, labels=labels)
    plt.xlabel("Log-marginal Likelihood")
    plt.title("Box Plot of Log-marginal Likelihood Estimates with Different Particle Settings")
    plt.grid()
    if figname:
        plt.savefig(f"report/figs/box_plot_{figname}.jpg")
    plt.show()

def plot_particle_convergence(pf_results, log_likelihood_kf, figname = None):
    num_particles = [key[0] for key in pf_results.keys()]
    mean_log_likelihoods = []
    for key in pf_results.keys():
        logs = np.zeros(len(pf_results[key]))
        for i, run in enumerate(pf_results[key]):
            logs[i] = run[1]
        mean_log_likelihoods.append(np.mean(logs))

    plt.figure(figsize=(10, 6))
    plt.scatter(num_particles, mean_log_likelihoods, label="Particle Filter Log Likelihoods", color="blue")
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

def plot_particle_loglog(pf_results, x_kalman, particles = True, mse=True, figname = None):
    if particles:
        num = [key[0] for key in pf_results.keys()]
        plot_type = "particles"
    else:
        num = [key[1] for key in pf_results.keys()]
        plot_type = "runs"
    data = []
    for key in pf_results.keys():
        res = np.zeros((len(pf_results[key]), len(pf_results[key][0][0]), 2))
        for i, run in enumerate(pf_results[key]):
            res[i] = run[0]
        data.append(np.mean(res, axis=0))
    if mse:
        errors = [np.sum((run - x_kalman)**2) / len(run) for run in data]
        error_type = "MSE"
    else:
        errors = [np.linalg.norm(run - x_kalman) for run in data]
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


def plot_runs_loglog(log_likelihoods_pf, x_kalman, mse=True):
    num_runs = [key[1] for key in log_likelihoods_pf.keys()]

    if mse:
        errors = [np.sum((run - x_kalman)**2) / len(run) for run in log_likelihoods_pf.values()]
    else:
        errors = [np.linalg.norm(run - x_kalman) for run in log_likelihoods_pf.values()]
    

    plt.figure(figsize=(10, 6))
    plt.loglog(num_runs, errors, marker='o', label="L2 error")

    slope, icpt = np.polyfit(np.log(num_runs), np.log(errors), 1)

    plt.loglog(num_runs, np.exp(icpt) * np.array(num_runs)**slope, linestyle='--', label=f"slope = {slope:.2f}")

    plt.xlabel("Number of runs")
    plt.ylabel("L2 Error")
    plt.legend()
    plt.grid(True)
    plt.show()