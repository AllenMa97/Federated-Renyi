import torch
import numpy as np

from tool.logger import *


def D_hat_θ(client_dataset, mask_s1_flag, client_model, criterion, device, hypothesis):
    client_X = client_dataset["X"]
    client_y = client_dataset["y"]
    if mask_s1_flag:
        client_s = client_dataset["s2"]
    else:
        client_s = client_dataset["s1"]

    a, b = 1, 0

    c0, c1 = (client_y == 0), (client_y == 1)
    sa, sb = (client_s == a), (client_s == b)

    sa_c0, sa_c1 = sa * c0, sa * c1
    sb_c0, sb_c1 = sb * c0, sb * c1

    m_sa_c0, m_sa_c1 = sum(sa_c0), sum(sa_c1)
    m_sb_c0, m_sb_c1 = sum(sb_c0), sum(sb_c1)

    X_sa_c0, X_sa_c1, y_sa_c0, y_sa_c1 = [], [], [], []
    for index, item in enumerate(sa_c0):
        if item:
            X_sa_c0.append(client_X[index])
            y_sa_c0.append(client_y[index])
        else:
            X_sa_c1.append(client_X[index])
            y_sa_c1.append(client_y[index])
    X_sa_c0, X_sa_c1, y_sa_c0, y_sa_c1 = torch.tensor(np.array(X_sa_c0)).to(device), torch.tensor(np.array(X_sa_c1)).to(
        device), \
                                         torch.tensor(np.array(y_sa_c0)).to(device), torch.tensor(np.array(y_sa_c1)).to(
        device)

    X_sb_c0, X_sb_c1, y_sb_c0, y_sb_c1 = [], [], [], []
    for index, item in enumerate(sa_c0):
        if item:
            X_sb_c0.append(client_X[index])
            y_sb_c0.append(client_y[index])
        else:
            X_sb_c1.append(client_X[index])
            y_sb_c1.append(client_y[index])
    X_sb_c0, X_sb_c1, y_sb_c0, y_sb_c1 = torch.tensor(np.array(X_sb_c0)).to(device), torch.tensor(np.array(X_sb_c1)).to(
        device), \
                                         torch.tensor(np.array(y_sb_c0)).to(device), torch.tensor(np.array(y_sb_c1)).to(
        device)

    L_hat_ac0, L_hat_ac1 = 0, 0
    for X, y in zip(X_sa_c0, y_sa_c0):
        prediction = client_model(X).to(device)
        if "LR" in hypothesis:
            loss = criterion(prediction, y.reshape(-1).float())
        else:
            loss = criterion(prediction, y.long())
        L_hat_ac0 += loss * 1 / m_sa_c0

    for X, y in zip(X_sa_c1, y_sa_c1):
        prediction = client_model(X).to(device)
        if "LR" in hypothesis:
            loss = criterion(prediction, y.reshape(-1).float())
        else:
            loss = criterion(prediction, y.long())
        L_hat_ac1 += loss * 1 / m_sa_c1

    L_hat_bc0, L_hat_bc1 = 0, 0
    for X, y in zip(X_sb_c0, y_sb_c0):
        prediction = client_model(X).to(device)
        if "LR" in hypothesis:
            loss = criterion(prediction, y.reshape(-1).float())
        else:
            loss = criterion(prediction, y.long())
        L_hat_ac0 += loss * 1 / m_sb_c0
    for X, y in zip(X_sb_c1, y_sb_c1):
        prediction = client_model(X).to(device)
        if "LR" in hypothesis:
            loss = criterion(prediction, y.reshape(-1).float())
        else:
            loss = criterion(prediction, y.long())
        L_hat_ac1 += loss * 1 / m_sb_c1

    L_hat_ac = L_hat_ac0 + L_hat_ac1
    L_hat_bc = L_hat_bc0 + L_hat_bc1
    return L_hat_ac - L_hat_bc


def LCO_LR(device,
           global_model,
           algorithm_epoch_T, num_clients_N,
           training_dataloaders,
           training_dataset,
           client_dataset_list,
           ϵ
           ):
    client_datasets_size_list = []
    for i in range(num_clients_N):
        client_datasets_indices = client_dataset_list[i].indices
        client_datasets_dict = training_dataset[client_datasets_indices]
        client_datasets_size_list.append(len(client_datasets_indices))
        client_dataset_list[i] = client_datasets_dict

    m_i_list = torch.tensor(client_datasets_size_list)
    m_total = sum(client_datasets_size_list)

    # Training process
    logger.info("Training process")
    # Parameter Initialization

    α = 0.05
    β = 0.05
    γ = 0.001

    criterion = torch.nn.BCELoss()

    for iter_t in range(algorithm_epoch_T):
        if iter_t == 0:
            λ_a_list = [0.1] * num_clients_N
            λ_b_list = [0.1] * num_clients_N
        if iter_t % 20000 == 0 and iter_t != 0:
            α = 0.1 * α
        client_loss_list = []
        client_D_hat_list = []

        # Simulate Client Parallel for computation
        for i in range(num_clients_N):
            logger.info(f"########## Algorithm Epoch: {iter_t + 1} / {algorithm_epoch_T}; "
                        f"Client: {i + 1} / {num_clients_N};  ##########")

            client_i_dataloader = training_dataloaders[i]
            client_dataset = client_dataset_list[i]
            # Since there is no local update in Algorithm 1 in paper, so local model is always equal to global model
            client_model = global_model.to(device)
            client_model.train()
            client_i_loss = 0

            for batch_index, batch in enumerate(client_i_dataloader):
                X = batch["X"].to(device)
                y = batch["y"].reshape(-1, 1).to(device)
                local_prediction = client_model(X).to(device)
                loss = criterion(local_prediction, y.float())
                client_i_loss += loss

            # L_hat_i_θ in Eq.4
            client_i_loss = client_i_loss / m_i_list[i]
            D_hat_i_θ = D_hat_θ(client_dataset, False, client_model, criterion, device, "LR")

            # Equation 11
            λ_a = λ_a_list[i]
            λ_b = λ_b_list[i]
            first_term_in_Eq11 = (m_i_list[i] / m_total) * client_i_loss
            second_term_in_Eq11 = (λ_a - λ_b) * D_hat_i_θ
            client_i_loss = first_term_in_Eq11 + second_term_in_Eq11

            client_loss_list.append(client_i_loss)
            client_D_hat_list.append(D_hat_i_θ)

        # Parameter update by Equation 10
        optimizer = torch.optim.SGD(global_model.parameters(), lr=α)
        global_loss = sum(client_loss_list)
        global_loss.backward()
        optimizer.step()

        # Equation 12 & 13
        common_term_of_the_first_term_in_Eq12_Eq13 = 1 - γ * β
        first_term_in_Eq12 = common_term_of_the_first_term_in_Eq12_Eq13 * λ_a
        first_term_in_Eq13 = common_term_of_the_first_term_in_Eq12_Eq13 * λ_b

        second_term_in_Eq12_Eq_13 = β * torch.tensor(client_D_hat_list)

        third_term_in_Eq12_Eq_13 = β * ϵ

        eq_12 = first_term_in_Eq12 + second_term_in_Eq12_Eq_13 - third_term_in_Eq12_Eq_13
        eq_13 = first_term_in_Eq13 - second_term_in_Eq12_Eq_13 - third_term_in_Eq12_Eq_13

        # Updates λ_a, λ_b by Equation 12 & 13
        for i in range(len(λ_a_list)):
            λ_a_list[i] = max(eq_12[i], 0)
            λ_b_list[i] = max(eq_13[i], 0)

    logger.info("Training finish, return global model")
    return global_model


def LCO_NN(device,
           global_model,
           algorithm_epoch_T, num_clients_N,
           training_dataloaders,
           training_dataset,
           client_dataset_list,
           ϵ
           ):
    client_datasets_size_list = []
    for i in range(num_clients_N):
        client_datasets_indices = client_dataset_list[i].indices
        client_datasets_dict = training_dataset[client_datasets_indices]
        client_datasets_size_list.append(len(client_datasets_indices))
        client_dataset_list[i] = client_datasets_dict

    m_i_list = torch.tensor(client_datasets_size_list)
    m_total = sum(client_datasets_size_list)

    # Training process
    logger.info("Training process")
    # Parameter Initialization

    α = 0.05
    β = 0.05
    γ = 0.001

    criterion = torch.nn.CrossEntropyLoss()

    for iter_t in range(algorithm_epoch_T):
        if iter_t == 0:
            λ_a_list = [0.1] * num_clients_N
            λ_b_list = [0.1] * num_clients_N
        if iter_t % 20000 == 0 and iter_t != 0:
            α = 0.1 * α
        client_loss_list = []
        client_D_hat_list = []

        # Simulate Client Parallel for computation
        for i in range(num_clients_N):
            logger.info(f"########## Algorithm Epoch: {iter_t + 1} / {algorithm_epoch_T}; "
                        f"Client: {i + 1} / {num_clients_N};  ##########")

            client_i_dataloader = training_dataloaders[i]
            client_dataset = client_dataset_list[i]
            # Since there is no local update in Algorithm 1 in paper, so local model is always equal to global model
            client_model = global_model.to(device)
            client_model.train()
            client_i_loss = 0

            for batch_index, batch in enumerate(client_i_dataloader):
                X = batch["X"].to(device)
                y = batch["y"].to(device)
                local_prediction = client_model(X).to(device)
                loss = criterion(local_prediction, y.long())
                client_i_loss += loss

            # L_hat_i_θ in Eq.4
            client_i_loss = client_i_loss / m_i_list[i]
            D_hat_i_θ = D_hat_θ(client_dataset, False, client_model, criterion, device, "NN")

            # Equation 11
            λ_a = λ_a_list[i]
            λ_b = λ_b_list[i]
            first_term_in_Eq11 = (m_i_list[i] / m_total) * client_i_loss
            second_term_in_Eq11 = (λ_a - λ_b) * D_hat_i_θ

            client_i_loss = first_term_in_Eq11 + second_term_in_Eq11

            client_loss_list.append(client_i_loss)
            client_D_hat_list.append(D_hat_i_θ)

        # Parameter update by Equation 10
        optimizer = torch.optim.SGD(global_model.parameters(), lr=α)
        global_loss = sum(client_loss_list)
        global_loss.backward()
        optimizer.step()

        # Equation 12 & 13
        common_term_of_the_first_term_in_Eq12_Eq13 = 1 - γ * β
        first_term_in_Eq12 = common_term_of_the_first_term_in_Eq12_Eq13 * λ_a
        first_term_in_Eq13 = common_term_of_the_first_term_in_Eq12_Eq13 * λ_b

        second_term_in_Eq12_Eq_13 = β * torch.tensor(client_D_hat_list)

        third_term_in_Eq12_Eq_13 = β * ϵ

        eq_12 = first_term_in_Eq12 + second_term_in_Eq12_Eq_13 - third_term_in_Eq12_Eq_13
        eq_13 = first_term_in_Eq13 - second_term_in_Eq12_Eq_13 - third_term_in_Eq12_Eq_13

        # Updates λ_a, λ_b by Equation 12 & 13
        for i in range(len(λ_a_list)):
            λ_a_list[i] = max(eq_12[i], 0)
            λ_b_list[i] = max(eq_13[i], 0)

    logger.info("Training finish, return global model")
    return global_model
