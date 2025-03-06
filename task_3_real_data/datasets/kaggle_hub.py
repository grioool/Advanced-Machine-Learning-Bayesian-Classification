import kagglehub

def download_dataset_1():
    path = kagglehub.dataset_download("mlg-ulb/creditcardfraud")
    print("Path to dataset files:", path)

def download_dataset_2():
    path = kagglehub.dataset_download("ritesaluja/bank-note-authentication-uci-data")
    print("Path to dataset files:", path)

def download_dataset_3():
    path = kagglehub.dataset_download("uciml/pima-indians-diabetes-database")
    print("Path to dataset files:", path)