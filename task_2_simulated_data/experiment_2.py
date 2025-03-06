import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

from task_2_simulated_data.data_generation import generate_scheme1, generate_scheme2, evaluate_methods


class Experiment2:

    def __init__(self, n, a, rho_values, n_repeats):
        self.n = n
        self.a = a  # fixed value a = 2
        self.rho_values = rho_values
        self.n_repeats = n_repeats
        self.methods = ['LDA', 'QDA', 'NB']
        self.results_scheme1 = {method: {rho: [] for rho in rho_values} for method in self.methods}
        self.results_scheme2 = {method: {rho: [] for rho in rho_values} for method in self.methods}

    def run(self):
        for rho in self.rho_values:
            for rep in range(self.n_repeats):
                X_train, y_train = generate_scheme1(self.n, self.a)
                X_test, y_test = generate_scheme1(self.n, self.a)
                acc_dict = evaluate_methods(X_train, y_train, X_test, y_test)
                for method in self.methods:
                    self.results_scheme1[method][rho].append(acc_dict[method])

                X_train, y_train = generate_scheme2(self.n, self.a, rho)
                X_test, y_test = generate_scheme2(self.n, self.a, rho)
                acc_dict = evaluate_methods(X_train, y_train, X_test, y_test)
                for method in self.methods:
                    self.results_scheme2[method][rho].append(acc_dict[method])
        print("Experiment 2 completed.")

    def plot_results(self, filename):
        offsets = {'LDA': -0.2, 'QDA': 0, 'NB': 0.2}
        group_centers = np.arange(1, len(self.rho_values) + 1)

        with PdfPages(filename) as pdf:
            fig, ax = plt.subplots(figsize=(10, 6))
            boxplot_data = []
            positions = []
            for i, rho in enumerate(self.rho_values):
                center = group_centers[i]
                for method in self.methods:
                    boxplot_data.append(self.results_scheme1[method][rho])
                    positions.append(center + offsets[method])
            ax.boxplot(boxplot_data, positions=positions, widths=0.15, patch_artist=True)
            ax.set_xticks(group_centers)
            ax.set_xticklabels(self.rho_values)
            ax.set_xlabel("ρ values")
            ax.set_ylabel("Accuracy")
            ax.set_title("Scheme 1 - Independent Features")

            for method, offset in offsets.items():
                ax.plot([], [], label=method)
            ax.legend()
            pdf.savefig(fig)
            plt.close()

            fig, ax = plt.subplots(figsize=(10, 6))
            boxplot_data = []
            positions = []
            for i, rho in enumerate(self.rho_values):
                center = group_centers[i]
                for method in self.methods:
                    boxplot_data.append(self.results_scheme2[method][rho])
                    positions.append(center + offsets[method])
            ax.boxplot(boxplot_data, positions=positions, widths=0.15, patch_artist=True)
            ax.set_xticks(group_centers)
            ax.set_xticklabels(self.rho_values)
            ax.set_xlabel("ρ values")
            ax.set_ylabel("Accuracy")
            ax.set_title("Scheme 2 - Correlated Features")
            for method, offset in offsets.items():
                ax.plot([], [], label=method)
            ax.legend()
            pdf.savefig(fig)
            plt.close()
        print(f"Results saved in '{filename}'.")

    def visualize_datasets(self, rho):
        X1, y1 = generate_scheme1(self.n, self.a)
        X2, y2 = generate_scheme2(self.n, self.a, rho)

        fig, axes = plt.subplots(1, 2, figsize=(12, 6))

        axes[0].scatter(X1[y1 == 0][:, 0], X1[y1 == 0][:, 1], c='blue', label='Class 0', alpha=0.5)
        axes[0].scatter(X1[y1 == 1][:, 0], X1[y1 == 1][:, 1], c='red', label='Class 1', alpha=0.5)
        axes[0].set_title(f"Scheme 1 (a = {self.a})")
        axes[0].set_xlabel("Feature 1")
        axes[0].set_ylabel("Feature 2")
        axes[0].legend()

        axes[1].scatter(X2[y2 == 0][:, 0], X2[y2 == 0][:, 1], c='blue', label='Class 0', alpha=0.5)
        axes[1].scatter(X2[y2 == 1][:, 0], X2[y2 == 1][:, 1], c='red', label='Class 1', alpha=0.5)
        axes[1].set_title(f"Scheme 2 (a = {self.a}, ρ = {rho})")
        axes[1].set_xlabel("Feature 1")
        axes[1].set_ylabel("Feature 2")
        axes[1].legend()

        plt.tight_layout()
        plt.show()
