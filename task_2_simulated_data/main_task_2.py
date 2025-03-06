from task_2_simulated_data.data_generation import generate_scheme1, generate_scheme2, evaluate_methods
from task_2_simulated_data.experiment_1 import Experiment1
from task_2_simulated_data.experiment_2 import Experiment2
from task_2_simulated_data.experiment_3 import Experiment3


def test_generated_data():
    n = 1000
    a = 1.0
    X_train, y_train = generate_scheme1(n, a)
    X_test, y_test = generate_scheme1(n, a)
    print("Scheme 1 accuracies:", evaluate_methods(X_train, y_train, X_test, y_test))

    rho = 0.5
    X_train, y_train = generate_scheme2(n, a, rho)
    X_test, y_test = generate_scheme2(n, a, rho)
    print("Scheme 2 accuracies:", evaluate_methods(X_train, y_train, X_test, y_test))

def experiement_1():
    experiment = Experiment1(n=1000, a_values=[0.1, 0.5, 1, 2, 3, 5], rho=0.5, n_repeats=50)
    experiment.run()
    experiment.plot_results("BayesianSimulatedData1.pdf")
    experiment.visualize_datasets(a=1)

def experiement_2():
    experiment2 = Experiment2(n=1000, a=2, rho_values=[0, 0.1, 0.3, 0.5, 0.7, 0.9], n_repeats=50)
    experiment2.run()
    experiment2.plot_results("BayesianSimulatedData2.pdf")
    experiment2.visualize_datasets(rho=0.5)

def experiement_3():
    experiment3 = Experiment3(n=1000, a=2, rho=0.5)
    experiment3.run()

if __name__ == "__main__":
    test_generated_data()
    experiement_1()
    experiement_2()
    experiement_3()
