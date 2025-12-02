from svm_dem import svm
from decisiontree_dem import decisiontree
from pathlib import Path
from grid_dem import demodulate
from torchvision import transforms
from optuna.trial import TrialState
from dataloader16gb import dataloader16gb
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler

import torch.nn as nn
import pandas as pd
import numpy as np

import optuna
import torch
import os

database_path = "../3x16Gbaud_16QAM_B2B_250km"
batch_size = 100
parametrization_trials = 100
parametrization_time = 36000

MOD_DICT = {
    0: -3 + 3j,  # 0000
    1: -3 + 1j,  # 0001
    2: -3 - 3j,  # 0010
    3: -3 - 1j,  # 0011
    4: -1 + 3j,  # 0100
    5: -1 + 1j,  # 0101
    6: -1 - 3j,  # 0110
    7: -1 - 1j,  # 0111
    8: +3 + 3j,  # 1000
    9: +3 + 1j,  # 1001
    10: 3 - 3j,  # 1010
    11: 3 - 1j,  # 1011
    12: 1 + 3j,  # 1100
    13: 1 + 1j,  # 1101
    14: 1 - 3j,  # 1110
    15: 1 - 1j,  # 1111
}

spacing_list = ["15", "15p5", "16", "16p5", "17", "17p6", "18", "50"]

bernonequ = []
berequ = []
svmequ = []
svmnoequ = []
treeequ = []
treenoequ = []
names = []


class QAMDataset(Dataset):
    def __init__(self, features, targets):
        self.features = features
        self.targets = targets

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]


def sync_signals(tx: np.ndarray, rx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Synchronizes two signals.

    Parameters:
        tx: Short signal, usually the received signal.
        rx: Long signal, usually the transmitted signal.

    Returns:
        tuple[np.ndarray, np.ndarray]: Synchronized copies of both signals in the same order as the input parameters.
    """
    tx_long = np.concatenate((tx, tx))
    correlation = np.abs(
        np.correlate(
            np.abs(tx_long) - np.mean(np.abs(tx_long)),
            np.abs(rx) - np.mean(np.abs(rx)),
            mode="full",
        )
    )
    delay = np.argmax(correlation) - len(rx) + 1
    sync_signal = tx_long[delay:]
    sync_signal = sync_signal[: len(rx)]

    return sync_signal, rx


def translate_tx(tx_i: list, tx_q: list, MOD_DICT: dict) -> list:
    """
    translate two list of entire numbers (they represent an imaginary number) in to a list of symbols (defined by the dictionary)

    parameters:
        tx_q (list): the list of real parts for the imaginary numbers
        tx_i (list): the list of imaginary parts for the imaginary numbers
        MOD_DICT (dict): the dictionary that contains the translation guide

    returns:
        tx_real (list): the list of translated symbols
    """
    tx_real = []

    for i in range(0, len(tx_i)):
        for key, val in MOD_DICT.items():
            if val == tx_i[i] + tx_q[i] * 1j:
                tx_real.append(key)
    return tx_real


def objective(trial):
    DEVICE = torch.device("cuda")
    model = define_model(trial).to(torch.device(DEVICE))
    # criterion = nn.MSELoss()
    criterion = nn.L1Loss()
    # criterion = nn.HuberLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    EPOCHS = 100
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        for batch_features, batch_targets in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_features.cuda())
            loss = criterion(outputs.cuda(), batch_targets.cuda())
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * batch_features.size(0)

        # Calcular la pérdida promedio por epoch
        train_loss = train_loss / len(train_loader.dataset)

        # Validación
        model.eval()  # Poner el modelo en modo evaluación
        test_loss = 0.0
        with torch.no_grad():
            for batch_features, batch_targets in test_loader:
                outputs = model(batch_features.cuda())
                loss = criterion(outputs.cuda(), batch_targets.cuda())
                test_loss += loss.item() * batch_features.size(0)

        test_loss = test_loss / len(test_loader.dataset)
    return test_loss


def define_model(trial):
    # We optimize the number of layers, hidden units and dropout ratio in each layer.
    n_layers = trial.suggest_int("n_layers", 2, 6)
    layers = []
    layersre = []
    in_features = 2
    out_features = 196
    for i in range(n_layers):
        out_features = trial.suggest_int(
            "n_units_l{}".format(i), 4, out_features - 16, step=16
        )
        layers.append(nn.Linear(in_features, out_features))
        layers.append(nn.ReLU())
        layersre.append([out_features, in_features])
        in_features = out_features
        if out_features <= 16 and i != n_layers:
            out_features += 16
    if in_features > 16:
        layers.append(nn.Linear(in_features, 16))
        layers.append(nn.ReLU())
        layersre.append([16, in_features])
    for i in range(len(layersre) - 1, 0, -1):
        layers.append(nn.Linear(layersre[i][0], layersre[i][1]))
        layers.append(nn.ReLU())
        in_features = layersre[i][1]
    layers.append(nn.Linear(in_features, 2))
    return nn.Sequential(*layers)


def repeate_model(trial):
    n_layers = trial.params["n_layers"]
    layers = []
    layersre = []
    in_features = 2
    for i in range(n_layers):
        out_features = trial.params["n_units_l{}".format(i)]
        layers.append(nn.Linear(in_features, out_features))
        layers.append(nn.ReLU())
        layersre.append([out_features, in_features])
        in_features = out_features
        if out_features <= 16 and i != n_layers:
            out_features += 16
    if in_features > 16:
        layers.append(nn.Linear(in_features, 16))
        layers.append(nn.ReLU())
        layersre.append([16, in_features])
    for i in range(len(layersre) - 1, 0, -1):
        layers.append(nn.Linear(layersre[i][0], layersre[i][1]))
        layers.append(nn.ReLU())
        in_features = layersre[i][1]
    layers.append(nn.Linear(in_features, 2))
    return nn.Sequential(*layers)


def bit_error_rate(sym_rx: np.ndarray, sym_tx: np.ndarray) -> float:
    """
    Calculates the bit error rate (BER).

    Parameters:
        sym_rx (np.ndarray): Vector of received symbols.
        sym_tx (np.ndarray): Vector of transmitted symbols.

    Returns:
        float: Bit error rate, the proportion of bit errors.
    """
    # Convert symbols to binary strings
    sym_rx_str = "".join([f"{int(sym):04b}" for sym in sym_rx])
    sym_tx_str = "".join([f"{int(sym):04b}" for sym in sym_tx])

    if len(sym_rx_str) < len(sym_tx_str):
        error = sum(sym_rx_str[i] != sym_tx_str[i] for i in range(len(sym_rx_str)))
    else:
        error = sum(sym_rx_str[i] != sym_tx_str[i] for i in range(len(sym_tx_str)))
    ber = error / len(sym_rx_str)
    return ber


def __do_backup(filename: str, n_backups: int = 0) -> None:
    """
    Perform backup rotation for a file, keeping a specified number of backups.

    Parameters:
        filename (str): The name of the file to create or overwrite.
        n_backups (int, optional): The number of backup files to keep. Defaults to 0.
            - **-1**: Skips backup entirely (no changes made).
            - **0**: Overwrites the existing file (no backups kept).
            - **Positive value**:
                - Rotates existing backups to keep no more than `n_backups` versions.
                - Creates a new backup of the original file (if it exists) with the highest index.

    Returns:
        None
    """

    # Function to get backup filenames
    def backup_filename(index):
        return f"{filename}.bak{index}"

    # Check for n_backups sentinel value (-1) to skip backup logic
    if n_backups == -1:
        logger.warn(f"Skipping backup for {filename} (n_backups=-1).")
        return

    # Backup logic for positive n_backups
    for i in range(n_backups, 0, -1):
        src = backup_filename(i - 1) if i - 1 > 0 else filename
        dst = backup_filename(i)
        os.rename(src, dst) if os.path.exists(src) else None


""" -Cargar datos de entrenamieno- """

X_test = np.empty([0, 2])
Y_test = np.empty([0, 2])
X_train = np.empty([0, 2])
Y_train = np.empty([0, 2])

""" --Importar datos-- """

for spacing in spacing_list:
    if spacing == "50":
        db = "31.3"
    else:
        db = "32"

    rx, tx = dataloader16gb(database_path, spacing, db)
    df = pd.DataFrame({"grid": demodulate(rx, MOD_DICT)})

    """ --sincronizar datos-- """

    tx_symbols = translate_tx(tx["I"], tx["Q"], MOD_DICT)
    tx_symbols, df["grid"] = sync_signals(tx_symbols, df["grid"])

    txi = []
    txq = []

    for i in tx_symbols:
        txq.append(MOD_DICT[i].real)
        txi.append(MOD_DICT[i].imag)

    df["txi"] = txi
    df["txq"] = txq

    """ --filtrar datos-- """

    X = rx.values
    Y = df[["txi", "txq"]].values

    X_train_incomplete, X_test_incomplete, Y_train_incomplete, Y_test_incomplete = (
        train_test_split(X, Y, test_size=90000, shuffle=True)
    )

    X_test = np.append(X_test, X_test_incomplete, axis=0)
    Y_test = np.append(Y_test, Y_test_incomplete, axis=0)
    X_train = np.append(X_train, X_train_incomplete, axis=0)
    Y_train = np.append(Y_train, Y_train_incomplete, axis=0)

""" -Parametrizar modelo- """

scaler = MinMaxScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
Y_train_scaled = scaler.transform(Y_train)
Y_test_scaled = scaler.transform(Y_test)

X_train_tensor = torch.FloatTensor(X_train_scaled)
Y_train_tensor = torch.FloatTensor(Y_train_scaled)
X_test_tensor = torch.FloatTensor(X_test_scaled)
Y_test_tensor = torch.FloatTensor(Y_test_scaled)

train_dataset = QAMDataset(X_train_tensor, Y_train_tensor)
test_dataset = QAMDataset(X_test_tensor, Y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=parametrization_trials, timeout=parametrization_time)

pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

trial = study.best_trial

""" -Entrenar modelo- """

model = repeate_model(trial)
criterion = nn.L1Loss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

EPOCHS = 100
for epoch in range(EPOCHS):
    model.train()
    train_loss = 0.0
    for batch_features, batch_targets in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_features.cuda())
        loss = criterion(outputs.cuda(), batch_targets.cuda())
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * batch_features.size(0)

    # Calcular la pérdida promedio por epoch
    train_loss = train_loss / len(train_loader.dataset)

torch.save(model, "../resultados/modelos/variacion_ghz.pth")
# model = torch.load("../resultados/modelos/variacion_ghz.pth", weights_only = False)
model.eval()

""" -Cargar datos- """

for SPACING in spacing_list:
    osnrlist = []
    for directory in sorted(Path(database_path).iterdir()):
        if directory.is_file() or directory.name[-3:] != "GHz":
            if directory.name == "single_ch":
                dir_name = "50GHz"
            elif directory.name == "2x16QAM_16GBd.csv":
                tx = pd.read_csv(directory)
                continue
            else:
                continue
        else:
            dir_name = directory.name
        if SPACING == dir_name[0 : dir_name.find("GHz")]:
            for subdir in sorted(directory.iterdir()):
                if subdir.is_dir():
                    continue
                osnrlist.append(
                    subdir.name[
                        subdir.name.find("consY") + len("consY") : subdir.name.find(
                            "dB"
                        )
                    ]
                )
                rx, tx = dataloader16gb(
                    database_path,
                    SPACING,
                    subdir.name[
                        subdir.name.find("consY") + len("consY") : subdir.name.find(
                            "dB"
                        )
                    ],
                )

                """ -Demodular- """
                """ --tradicional-- """

                df = pd.DataFrame({"grid": demodulate(rx, MOD_DICT)})

                """ --sincronizar datos-- """

                tx_symbols = translate_tx(tx["I"], tx["Q"], MOD_DICT)
                df["tx_symbols"], df["grid"] = sync_signals(tx_symbols, df["grid"])

                """ --SVM-- """

                df["svm"], df["svm_tested"] = svm(
                    rx[["I", "Q"]].values, df["tx_symbols"].values
                )

                """ --decision tree-- """

                df["tree"], df["tree_tested"] = decisiontree(
                    rx[["I", "Q"]].values, df["tx_symbols"].values
                )

                """ - calcular metricas- """

                bernonequ.append(bit_error_rate(df["grid"], df["tx_symbols"]))
                svmnoequ.append(bit_error_rate(df["svm"], df["svm_tested"]))
                treenoequ.append(bit_error_rate(df["tree"], df["tree_tested"]))

                """ -Ecualizar- """
                txi = []
                txq = []
                for i in df["tx_symbols"]:
                    txq.append(MOD_DICT[i].real)
                    txi.append(MOD_DICT[i].imag)
                tx = pd.DataFrame({"I": txi, "Q": txq})
                X_test = rx.values
                Y_test = tx.values
                scaler = MinMaxScaler().fit(X_test)
                X_test_s = scaler.transform(X_test)
                Y_test_s = scaler.transform(Y_test)
                X_test_tensor = torch.FloatTensor(X_test_s)
                Y_test_tensor = torch.FloatTensor(Y_test_s)
                test_dataset = QAMDataset(X_test_tensor, Y_test_tensor)
                test_loader = DataLoader(
                    test_dataset, batch_size=batch_size, shuffle=False
                )

                criterion = nn.L1Loss()
                eq_symbols = []
                model.eval()  # Poner el modelo en modo evaluación
                test_loss = 0.0
                x = []
                y = []
                with torch.no_grad():
                    for batch_features, batch_targets in test_loader:
                        outputs = model(batch_features.cuda())
                        eq_symbols.extend(outputs.numpy())
                        loss = criterion(outputs.cuda(), batch_targets.cuda())
                        test_loss += loss.item() * batch_features.size(0)
                        for i in range(len(outputs)):
                            x.append(outputs.numpy()[i][0])
                            y.append(outputs.numpy()[i][1])

                eq_symbols = np.array(eq_symbols)
                test_loss = test_loss / len(test_loader.dataset)
                eq_symbols_is = scaler.inverse_transform(eq_symbols)

                I = []
                Q = []
                for datapoint in eq_symbols_is:
                    I.append(datapoint[0])
                    Q.append(datapoint[1])
                df["I_equalized"] = I
                df["Q_equalized"] = Q

                """ -Demodular- """

                I = []
                Q = []
                for datapoint in Y_test:
                    I.append(datapoint[0])
                    Q.append(datapoint[1])

                tx = np.array(demodulate(pd.DataFrame({"I": I, "Q": Q}), MOD_DICT))

                """ --tradicional-- """
                dfasist = pd.DataFrame({"I": df["I_equalized"], "Q": df["Q_equalized"]})
                df["grid_equlized"] = demodulate(dfasist, MOD_DICT)

                """ --SVM-- """

                df["svm_equalized"], df["svm_equalized_tested"] = svm(
                    df[["I_equalized", "Q_equalized"]].values, tx
                )

                """ --decision tree-- """

                df["tree_equalized"], df["tree_equalized_tested"] = decisiontree(
                    df[["I_equalized", "Q_equalized"]].values, tx
                )

                """ -calcular metricas- """

                berequ.append(
                    bit_error_rate(
                        df["grid_equlized"],
                        demodulate(pd.DataFrame({"I": I, "Q": Q}), MOD_DICT),
                    )
                )
                svmequ.append(
                    bit_error_rate(df["svm_equalized"], df["svm_equalized_tested"])
                )
                treeequ.append(
                    bit_error_rate(df["tree_equalized"], df["tree_equalized_tested"])
                )

                """ -Guardar los resultados- """

                df.to_parquet(
                    path=f"../resultados/{SPACING}GHZ/{subdir.name[subdir.name.find('consY') + len('consY') : subdir.name.find('dB')]}dB.parquet",
                    index=False,
                )

                names.append(
                    f"{SPACING}GHZ {subdir.name[subdir.name.find('consY') + len('consY') : subdir.name.find('dB')]}dB"
                )
                diccion = {
                    "name": names,
                    "grid": bernonequ,
                    "grid_equ": berequ,
                    "svm": svmnoequ,
                    "svm_equ": svmequ,
                    "tree": treenoequ,
                    "tree_equ": treeequ,
                }

                __do_backup("../resultados/%ber_variacion_ghz.parqueth", 1)
                pd.DataFrame(diccion).to_parquet(
                    "../resultados/%ber_variacion_ghz.parqueth", index=False
                )
