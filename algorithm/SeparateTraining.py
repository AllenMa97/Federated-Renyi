import torch
import numpy as np
import copy

from tool.logger import *
from tool.utils import get_parameters, set_parameters


def ST_LR(global_model, algorithm_epoch_T, num_clients_m, training_dataloaders):
    # Training process
    logger.info("Training process")

    # Parameter Initialization
    global_model.train()
    local_model_list = [copy.deepcopy(global_model) for i in range(num_clients_m)]

    for iter_t in range(algorithm_epoch_T):
        # Simulate Client Parallel
        for i in range(num_clients_m):
            model = local_model_list[i]
            model.train()

            # local option
            client_i_dataloader = training_dataloaders[i]

            logger.info(f"########## Algorithm Epoch: {iter_t + 1} / {algorithm_epoch_T}; "
                        f"Client: {i + 1} / {num_clients_m};  ##########")
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

    logger.info("Training finish, return local model list")

    return local_model_list


def ST_NN(global_model, algorithm_epoch_T, num_clients_m, training_dataloaders):
    # Training process
    logger.info("Training process")

    # Parameter Initialization
    global_model.train()
    local_model_list = [copy.deepcopy(global_model) for i in range(num_clients_m)]

    criterion = torch.nn.CrossEntropyLoss()

    for iter_t in range(algorithm_epoch_T):
        # Simulate Client Parallel
        for i in range(num_clients_m):
            model = local_model_list[i]
            model.train()

            optimizer = torch.optim.SGD(model.parameters(), lr=local_step_size)

            # local option
            client_i_dataloader = training_dataloaders[i]

            optimizer.zero_grad()
            logger.info(f"########## Algorithm Epoch: {iter_t + 1} / {algorithm_epoch_T}; "
                        f"Client: {i + 1} / {num_clients_m};  ##########")
            for batch_index, batch in enumerate(client_i_dataloader):
                X = batch["X"]
                y = batch["y"]
                local_prediction = model(X)
                loss = criterion(local_prediction, y.long())
                # logger.info(f"########## Algorithm Epoch: {iter_t + 1} / {algorithm_epoch_T}; "
                #             f"Client: {i + 1} / {num_clients_m}; "
                #             f"Batch: {batch_index}; Loss: {loss.data} ##########")

                loss.backward()
                optimizer.step()

            # Upgrade the local model list
            local_model_list[i] = model

    logger.info("Training finish, return local model list")

    return local_model_list
