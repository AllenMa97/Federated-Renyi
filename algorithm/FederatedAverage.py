import torch
import numpy as np
import copy

from tool.logger import *
from tool.utils import get_parameters, set_parameters


def client_selection(client_num, fraction, dataset_size, client_dataset_size_list, drop_rate, style="FedAvg"):
    assert sum(client_dataset_size_list) == dataset_size
    idxs_users = [0]

    selected_num = max(int(fraction * client_num), 1)
    if float(drop_rate) != 0:
        drop_num = max(int(selected_num * drop_rate, 1))
        selected_num -= drop_num

    if style == "FedAvg":
        idxs_users = np.random.choice(
            a=range(client_num),
            size=selected_num,
            replace=False,
            p=[float(i / dataset_size) for i in client_dataset_size_list]
        )

    return idxs_users


# Federated Average with Logistic Regression Model
def Fed_AVG_LR(global_model,
               algorithm_epoch_T, num_clients_m, local_update_round_K, FL_fraction, FL_drop_rate, local_step_size,
               training_dataloaders,
               training_dataset,
               client_dataset_list
               ):
    training_dataset_size = len(training_dataset)
    client_datasets_size_list = [len(item) for item in client_dataset_list]

    # Training process
    logger.info("Training process")

    # Parameter Initialization
    global_model.train()
    local_model_list = [copy.deepcopy(global_model) for i in range(num_clients_m)]

    # Outer Round
    for iter_t in range(algorithm_epoch_T):
        # Simulate Client Parallel
        for i in range(num_clients_m):
            model = local_model_list[i]
            model.train()

            # local option
            client_i_dataloader = training_dataloaders[i]
            for iter_k in range(local_update_round_K):  # Inner Round
                logger.info(f"########## Algorithm Epoch: {iter_t + 1} / {algorithm_epoch_T}; "
                            f"Client: {i + 1} / {num_clients_m}; "
                            f"Local epoch:  {iter_k + 1} / {local_update_round_K} ##########")
                for batch_index, batch in enumerate(client_i_dataloader):
                    X = batch["X"]
                    y = batch["y"].reshape(-1, 1)
                    local_prediction = model(X).reshape(-1, 1)
                    local_theta = torch.from_numpy(get_parameters(model)[0])
                    local_theta_gradient = torch.mm(X.T, (local_prediction - y))  # Gradient of theta ([feature_size, 1])

                    local_theta -= (local_step_size * local_theta_gradient).T

                    model = set_parameters(model, [local_theta])

            # Upgrade the local model list
            local_model_list[i] = model

        # Client selection
        idxs_users = client_selection(
            client_num=num_clients_m,
            fraction=FL_fraction,
            dataset_size=training_dataset_size,
            client_dataset_size_list=client_datasets_size_list,
            drop_rate=FL_drop_rate,
            style="FedAvg",
        )
        logger.info(f"Select client list: {idxs_users} ")

        # Global operation
        logger.info("Parameter aggregation")
        theta_list = []
        for id in idxs_users:
            selected_model = local_model_list[id]
            theta_list.append(get_parameters(selected_model))
        theta_list = np.array(theta_list)
        theta_avg = np.mean(theta_list, 0).tolist()
        set_parameters(global_model, theta_avg)

        # Parameter Distribution
        logger.info("Parameter Distribution")
        local_model_list = [copy.deepcopy(global_model) for i in range(num_clients_m)]

    logger.info("Training finish, return global model")

    return global_model


def Fed_AVG_NN(global_model,
               algorithm_epoch_T, num_clients_m, local_update_round_K, FL_fraction, FL_drop_rate, local_step_size,
               training_dataloaders,
               training_dataset,
               client_dataset_list):
    training_dataset_size = len(training_dataset)
    client_datasets_size_list = [len(item) for item in client_dataset_list]

    # Training process
    logger.info("Training process")

    # Parameter Initialization
    global_model.train()
    local_model_list = [copy.deepcopy(global_model) for i in range(num_clients_m)]

    # Outer Round
    for iter_t in range(algorithm_epoch_T):
        # Simulate Client Parallel
        for i in range(num_clients_m):
            model = local_model_list[i]
            model.train()
            criterion = torch.nn.CrossEntropyLoss()
            optimizer = torch.optim.SGD(model.parameters(), lr=local_step_size)

            # local option
            client_i_dataloader = training_dataloaders[i]
            for iter_k in range(local_update_round_K):  # Inner Round
                optimizer.zero_grad()

                for batch_index, batch in enumerate(client_i_dataloader):
                    X = batch["X"]
                    y = batch["y"]
                    local_prediction = model(X)
                    loss = criterion(local_prediction, y.long())

                    loss.backward()
                    optimizer.step()

                logger.info(f"########## Algorithm Epoch: {iter_t + 1} / {algorithm_epoch_T}; "
                            f"Client: {i + 1} / {num_clients_m}; "
                            # f"Client: {i + 1}  acc: {round(acc_numerator/ acc_denominator, 5)}; "
                            f"Local epoch:  {iter_k + 1} / {local_update_round_K} ##########")

            # Upgrade the local model list
            local_model_list[i] = model

        # Client selection
        idxs_users = client_selection(
            client_num=num_clients_m,
            fraction=FL_fraction,
            dataset_size=training_dataset_size,
            client_dataset_size_list=client_datasets_size_list,
            drop_rate=FL_drop_rate,
            style="FedAvg",
        )
        logger.info(f"Select client list: {idxs_users} ")

        # Global operation
        logger.info("Parameter aggregation")
        theta_list = []
        for id in idxs_users:
            selected_model = local_model_list[id]
            theta_list.append(get_parameters(selected_model))
        theta_list = np.array(theta_list)
        theta_avg = np.mean(theta_list, 0).tolist()
        set_parameters(global_model, theta_avg)

        # Parameter Distribution
        logger.info("Parameter Distribution")
        local_model_list = [copy.deepcopy(global_model) for i in range(num_clients_m)]

    logger.info("Training finish, return global model")
    return global_model
