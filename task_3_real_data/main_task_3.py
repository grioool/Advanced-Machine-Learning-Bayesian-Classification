from task_3_real_data.experiment_4 import ExperimentReal


def experiment_on_real_data():
    datasets = [
        ("datasets/BankNote_Authentication.csv", "Pima Indians Diabetes Database"),
        ("datasets/diabetes.csv", "Diabetes"),
        ("datasets/creditcard.csv", "Credit Card Fraud Detection")
    ]
    experiment = ExperimentReal(datasets, n_repeats=50, test_size=0.3)
    experiment.run()
    experiment.plot_results("BayesianReal.pdf")


if __name__ == "__main__":
    experiment_on_real_data()
