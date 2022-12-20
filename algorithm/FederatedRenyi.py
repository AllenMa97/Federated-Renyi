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


def Fed_Renyi(global_model,
              algorithm_epoch_T, num_clients_m, local_update_round_K, FL_fraction, FL_drop_rate, local_step_size,
              training_dataloaders,
              training_dataset,
              client_dataset_list
            ):
    # Parameter Distribution
    logger.info("Parameter Distribution")
    local_model_list = [copy.deepcopy(global_model) for i in range(num_clients_m)]

    # Training process
    logger.info("Training process")
    global_model.train()

    training_dataset_size = len(training_dataset)
    client_datasets_size_list = [len(item) for item in client_dataset_list]

    # Parameter Initialization
    global_theta = get_parameters(global_model)
    global_v = 0
    lam = 75

    # Outer Round
    for iter_t in range(algorithm_epoch_T):
        # Simulate Client Parallel
        for i in range(num_clients_m):
            model = local_model_list[i]
            model.train()

            w = [0, 0]
            # Sensitive attribute 1
            # s = torch.tensor([training_dataset[idx]['s1'] for idx in client_dataset_list[i].indices])
            # Sensitive attribute 2
            # s = torch.tensor([training_dataset[idx]['s2'] for idx in client_dataset_list[i].indices])

            # local option
            client_i_dataloader = training_dataloaders[i]
            for iter_k in range(local_update_round_K):  # Inner Round
                logger.info(f"########## Algorithm Epoch: {iter_t+1} / {algorithm_epoch_T}; "
                            f"Client: {i+1} / {num_clients_m}; "
                            f"Local epoch:  {iter_k+1} / {local_update_round_K} ##########")
                for batch_index, batch in enumerate(client_i_dataloader):
                    X = batch["X"]
                    y = batch["y"].reshape(-1, 1)
                    s = batch["s1"]
                    local_prediction = model(X).reshape(-1, 1)
                    local_theta = torch.from_numpy(get_parameters(model)[0])
                    local_gradient_1 = torch.mm(X.T, (local_prediction - y))  # Gradient of theta ([feature_size, 1])

                    a = w[0] - w[1]
                    b = (s - w[1] * torch.ones_like(s)).reshape(-1, 1)  # [8, ]

                    sigm = local_prediction  # [8, 1]
                    sigm2 = torch.ones_like(sigm) - sigm  # [8, 1]

                    sigm_dev = torch.multiply(sigm, sigm2)  # [8, 1]

                    inner1 = a * local_prediction - b  # [8, 1]

                    inner = torch.multiply(inner1, sigm_dev)

                    final_res = torch.mm(a * X.T, inner)

                    local_gradient_2 = lam * final_res  # Gradient of v

                    local_theta -= (local_step_size * (local_gradient_1 - local_gradient_2)).T

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
        # col1 = sigmoid((np.dot(X, theta)))
        # col2 = np.ones_like(col1) - col1
        #
        # yhatT = np.concatenate((col1, col2)).reshape((2, -1))
        # yhat = yhatT.T
        #
        # inv = np.linalg.inv(np.dot(yhatT, yhat))
        # temp2 = np.dot(inv, yhatT)
        #
        # w = np.dot(temp2, s)
