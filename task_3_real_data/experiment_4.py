import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from LDA import LDA
from NB import NB
from QDA import QDA


class ExperimentReal:
    def __init__(self, datasets, n_repeats, test_size):
        self.datasets = datasets
        self.n_repeats = n_repeats
        self.test_size = test_size
        self.methods = ['LDA', 'QDA', 'NB']
        self.results = {}
        self.target_columns = {
            "datasets/BankNote_Authentication.csv": "class",
            "datasets/diabetes.csv": "Outcome",
            "datasets/creditcard.csv": "Class"
        }

    def run(self):
        for filename, dataset_name in self.datasets:
            print(f"Processing dataset: {dataset_name}")
            df = pd.read_csv(filename)
            target_col = self.target_columns.get(filename, df.columns[-1])

            X = df.drop(columns=[target_col]).values
            y = df[target_col].values

            self.results[dataset_name] = {method: [] for method in self.methods}

            for rep in range(self.n_repeats):
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=self.test_size, random_state=None)

                models = {
                    'LDA': LDA.LDA(),
                    'QDA': QDA.QDA(),
                    'NB': NB.NB()
                }
                for method, model in models.items():
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    acc = accuracy_score(y_test, y_pred)
                    self.results[dataset_name][method].append(acc)
            print(f"Completed dataset: {dataset_name}")

    def plot_results(self, filename):
        with PdfPages(filename) as pdf:
            for dataset_name, data in self.results.items():
                boxplot_data = [data[method] for method in self.methods]
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.boxplot(boxplot_data, tick_labels=self.methods, patch_artist=True)
                ax.set_title(f"Accuracy Comparison on {dataset_name}")
                ax.set_ylabel("Accuracy")
                pdf.savefig(fig)
                plt.close()
        print(f"Results saved in '{filename}'.")
