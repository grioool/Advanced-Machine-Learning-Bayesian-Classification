import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D

from LDA import LDA
from QDA import QDA
from task_2_simulated_data.data_generation import generate_scheme2


class Experiment3:
    def __init__(self, n, a, rho):
        self.n = n
        self.a = a
        self.rho = rho

    def run(self):
        X_train, y_train = generate_scheme2(self.n, self.a, self.rho)

        lda_model = LDA.LDA()
        qda_model = QDA.QDA()
        lda_model.fit(X_train, y_train)
        qda_model.fit(X_train, y_train)

        margin = 1
        x_min, x_max = X_train[:, 0].min() - margin, X_train[:, 0].max() + margin
        y_min, y_max = X_train[:, 1].min() - margin, X_train[:, 1].max() + margin
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                             np.linspace(y_min, y_max, 200))
        grid = np.c_[xx.ravel(), yy.ravel()]

        lda_probs = lda_model.predict_proba(grid).reshape(xx.shape)
        qda_probs = qda_model.predict_proba(grid).reshape(xx.shape)

        fig, ax = plt.subplots(figsize=(10, 8))

        ax.scatter(X_train[y_train == 0][:, 0], X_train[y_train == 0][:, 1],
                   color='blue', marker='o', label='Class 0', alpha=0.6)
        ax.scatter(X_train[y_train == 1][:, 0], X_train[y_train == 1][:, 1],
                   color='red', marker='x', label='Class 1', alpha=0.6)

        cs_lda = ax.contour(xx, yy, lda_probs, levels=[0.5],
                            colors='green', linestyles='dashed', linewidths=2)
        cs_qda = ax.contour(xx, yy, qda_probs, levels=[0.5],
                            colors='purple', linestyles='solid', linewidths=2)

        legend_elements = [Line2D([0], [0], color='green', lw=2, linestyle='dashed', label='LDA'),
                           Line2D([0], [0], color='purple', lw=2, linestyle='solid', label='QDA'),
                           Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=8,
                                  label='Class 0'),
                           Line2D([0], [0], marker='x', color='w', markerfacecolor='red', markersize=8,
                                  label='Class 1')]
        ax.legend(handles=legend_elements, loc='best')

        ax.set_xlabel("Feature 1")
        ax.set_ylabel("Feature 2")
        ax.set_title(f"a={self.a}, ρ={self.rho}")
        plt.tight_layout()

        with PdfPages("BayesianSimulatedData3.pdf") as pdf:
            pdf.savefig(fig)
        plt.close()
        print("Experiment 3 completed and saved in 'BayesianSimulatedData3.pdf'.")
