import torch
import numpy as np
import copy
from tool.logger import *
from tool.utils import get_parameters, set_parameters


device = "cuda" if torch.cuda.is_available() else "cpu"  # Get cpu or gpu device for experiment

def client_selection(client_num, fraction, dataset_size, client_dataset_size_list, drop_rate, style="FedAvg"):
    assert sum(client_dataset_size_list) == dataset_size
    idxs_users = [0]

    selected_num = max(int(fraction * client_num), 1)
    if float(drop_rate) != 0:
        drop_num = max(int(selected_num * drop_rate), 1)
        selected_num -= drop_num

    if style == "FedAvg":
        idxs_users = np.random.choice(
            a=range(client_num),
            size=selected_num,
            replace=False,
            p=[float(i / dataset_size) for i in client_dataset_size_list]
        )

    return idxs_users


def r_hat_p_initialization(num_clients_K, mask_s1_flag, training_dataset, client_dataset_list, p):
    r_bar_k_p_list = []
    for k in range(num_clients_K):
        if mask_s1_flag:
            # Sensitive attribute 2
            client_s = torch.tensor([training_dataset[idx]['s2'] for idx in client_dataset_list[k].indices])
        else:
            # Sensitive attribute 1
            client_s = torch.tensor([training_dataset[idx]['s1'] for idx in client_dataset_list[k].indices])
        r_bar_k_p = sum(client_s == p) / len(client_s)
        r_bar_k_p_list.append(r_bar_k_p)

    r_bar_k_p_list = np.array(r_bar_k_p_list)
    r_hat_p = r_bar_k_p_list.mean()
    return r_hat_p


def q_c_p(y_hat_θ, s, c, p):
    y_hat_θ_c = (y_hat_θ == c).to(device)
    s_p = (s == p).to(device)
    joint = (y_hat_θ_c * s_p).to(device)

    P_y_hat_θ_c = (sum(y_hat_θ_c) / len(y_hat_θ)).to(device)  # u^bar_k(c)
    P_s_p = (sum(s_p) / len(s)).to(device)  # r^bar_k(p)

    P_joint = (sum(joint) / len(s)).to(device)
    P_conditional = (P_joint / P_s_p).to(device)  # j^bar_k(c, p)

    if P_conditional == 0 or P_s_p == 0 or P_y_hat_θ_c == 0:
        q = torch.tensor(0.)
    else:
        q = P_conditional * P_s_p / torch.sqrt(P_y_hat_θ_c * P_s_p)

    return q.to(device)


def G_θ_v(y_hat_θ, s, v):
    q_00 = q_c_p(y_hat_θ, s, 0, 0)
    q_01 = q_c_p(y_hat_θ, s, 0, 1)
    q_10 = q_c_p(y_hat_θ, s, 1, 0)
    q_11 = q_c_p(y_hat_θ, s, 1, 1)
    Q = torch.tensor([
        [q_00, q_01],
        [q_10, q_11]
    ]).to(device)
    v = v.reshape(-1, 1).to(device)
    g = v.T.matmul(Q.T).matmul(Q).matmul(v)
    g = g[0][0].to(device)
    return g


def argmax_v_LR(idxs_users, local_model_list, mask_s1_flag, training_dataset, client_dataset_list, r_hat_p0, r_hat_p1, γ_k_style):
    training_dataset_size = len(training_dataset)

    j_bar_c0_p0_list = []
    j_bar_c0_p1_list = []
    j_bar_c1_p0_list = []
    j_bar_c1_p1_list = []
    u_bar_c0_list = []
    u_bar_c1_list = []

    for id in idxs_users:
        selected_model = local_model_list[id].to(device)

        if mask_s1_flag:
            # Sensitive attribute 2
            client_s = torch.tensor([training_dataset[idx]['s2'] for idx in client_dataset_list[id].indices]).to(device)
        else:
            # Sensitive attribute 1
            client_s = torch.tensor([training_dataset[idx]['s1'] for idx in client_dataset_list[id].indices])
        client_X = torch.tensor(np.array([training_dataset[idx]['X'] for idx in client_dataset_list[id].indices])).to(device)

        y_hat_θ = (selected_model(client_X) >= 0.5).reshape(-1).to(device)

        y_hat_θ_c0 = (y_hat_θ == 0).to(device)
        y_hat_θ_c1 = (y_hat_θ == 1).to(device)

        s_p0 = (client_s == 0).to(device)
        s_p1 = (client_s == 1).to(device)

        joint_c0_p0 = (y_hat_θ_c0 * s_p0).to(device)
        joint_c0_p1 = (y_hat_θ_c0 * s_p1).to(device)
        joint_c1_p0 = (y_hat_θ_c1 * s_p0).to(device)
        joint_c1_p1 = (y_hat_θ_c1 * s_p1).to(device)

        P_s_p0 = (sum(s_p0) / len(client_s)).to(device)
        P_s_p1 = (sum(s_p1) / len(client_s)).to(device)

        P_joint_c0_p0 = (sum(joint_c0_p0) / len(client_s)).to(device)
        P_joint_c0_p1 = (sum(joint_c0_p1) / len(client_s)).to(device)
        P_joint_c1_p0 = (sum(joint_c1_p0) / len(client_s)).to(device)
        P_joint_c1_p1 = (sum(joint_c1_p1) / len(client_s)).to(device)

        if "uniform_client" in γ_k_style:
            γ_k = 1 / len(idxs_users)
        else:
            γ_k = len(client_X) / training_dataset_size

        j_bar_c0_p0 = (γ_k * P_joint_c0_p0 / P_s_p0).to(device)
        j_bar_c0_p1 = (γ_k * P_joint_c0_p1 / P_s_p1).to(device)
        j_bar_c1_p0 = (γ_k * P_joint_c1_p0 / P_s_p0).to(device)
        j_bar_c1_p1 = (γ_k * P_joint_c1_p1 / P_s_p1).to(device)

        j_bar_c0_p0_list.append(j_bar_c0_p0)
        j_bar_c0_p1_list.append(j_bar_c0_p1)
        j_bar_c1_p0_list.append(j_bar_c1_p0)
        j_bar_c1_p1_list.append(j_bar_c1_p1)

        u_bar_c0 = (γ_k * sum(y_hat_θ_c0) / len(y_hat_θ)).to(device)
        u_bar_c1 = (γ_k * sum(y_hat_θ_c1) / len(y_hat_θ)).to(device)  # u^bar_k(c)
        u_bar_c0_list.append(u_bar_c0)
        u_bar_c1_list.append(u_bar_c1)


    j_hat_c0_p0 = torch.tensor(j_bar_c0_p0_list).sum()
    j_hat_c0_p1 = torch.tensor(j_bar_c0_p1_list).sum()
    j_hat_c1_p0 = torch.tensor(j_bar_c1_p0_list).sum()
    j_hat_c1_p1 = torch.tensor(j_bar_c1_p1_list).sum()
    u_hat_c0 = torch.tensor(u_bar_c0_list).sum()
    u_hat_c1 = torch.tensor(u_bar_c1_list).sum()

    q_hat_00 = j_hat_c0_p0 * r_hat_p0 / torch.sqrt(u_hat_c0 * r_hat_p0)
    q_hat_01 = j_hat_c0_p1 * r_hat_p1 / torch.sqrt(u_hat_c0 * r_hat_p1)
    q_hat_10 = j_hat_c1_p0 * r_hat_p0 / torch.sqrt(u_hat_c1 * r_hat_p0)
    q_hat_11 = j_hat_c1_p1 * r_hat_p1 / torch.sqrt(u_hat_c1 * r_hat_p1)
    Q_hat = torch.tensor([
        [q_hat_00, q_hat_01],
        [q_hat_10, q_hat_11]
    ]).to(device)

    u, s, v = torch.svd(Q_hat)

    second_singular_vector_of_Q_hat = v[1].reshape(-1, 1).to(device)
    return second_singular_vector_of_Q_hat


def argmax_v_NN(idxs_users, local_model_list, mask_s1_flag, training_dataset, client_dataset_list, r_hat_p0, r_hat_p1, γ_k_style):
    training_dataset_size = len(training_dataset)

    j_bar_c0_p0_list = []
    j_bar_c0_p1_list = []
    j_bar_c1_p0_list = []
    j_bar_c1_p1_list = []
    u_bar_c0_list = []
    u_bar_c1_list = []

    for id in idxs_users:
        selected_model = local_model_list[id]

        if mask_s1_flag:
            # Sensitive attribute 2
            client_s = torch.tensor([training_dataset[idx]['s2'] for idx in client_dataset_list[id].indices]).to(device)
        else:
            # Sensitive attribute 1
            client_s = torch.tensor([training_dataset[idx]['s1'] for idx in client_dataset_list[id].indices]).to(device)
        client_X = torch.tensor(np.array([training_dataset[idx]['X'] for idx in client_dataset_list[id].indices])).to(device)

        y_hat_θ = selected_model(client_X).argmax(dim=1).to(device)

        y_hat_θ_c0 = (y_hat_θ == 0).to(device)
        y_hat_θ_c1 = (y_hat_θ == 1).to(device)

        s_p0 = (client_s == 0).to(device)
        s_p1 = (client_s == 1).to(device)

        joint_c0_p0 = (y_hat_θ_c0 * s_p0).to(device)
        joint_c0_p1 = (y_hat_θ_c0 * s_p1).to(device)
        joint_c1_p0 = (y_hat_θ_c1 * s_p0).to(device)
        joint_c1_p1 = (y_hat_θ_c1 * s_p1).to(device)

        P_s_p0 = (sum(s_p0) / len(client_s)).to(device)
        P_s_p1 = (sum(s_p1) / len(client_s)).to(device)

        P_joint_c0_p0 = (sum(joint_c0_p0) / len(client_s)).to(device)
        P_joint_c0_p1 = (sum(joint_c0_p1) / len(client_s)).to(device)
        P_joint_c1_p0 = (sum(joint_c1_p0) / len(client_s)).to(device)
        P_joint_c1_p1 = (sum(joint_c1_p1) / len(client_s)).to(device)

        if "uniform_client" in γ_k_style:
            γ_k = 1 / len(idxs_users)
        else:
            γ_k = len(client_X) / training_dataset_size

        j_bar_c0_p0 = (γ_k * P_joint_c0_p0 / P_s_p0).to(device)
        j_bar_c0_p1 = (γ_k * P_joint_c0_p1 / P_s_p1).to(device)
        j_bar_c1_p0 = (γ_k * P_joint_c1_p0 / P_s_p0).to(device)
        j_bar_c1_p1 = (γ_k * P_joint_c1_p1 / P_s_p1).to(device)

        j_bar_c0_p0_list.append(j_bar_c0_p0)
        j_bar_c0_p1_list.append(j_bar_c0_p1)
        j_bar_c1_p0_list.append(j_bar_c1_p0)
        j_bar_c1_p1_list.append(j_bar_c1_p1)

        u_bar_c0 = (γ_k * sum(y_hat_θ_c0) / len(y_hat_θ)).to(device)
        u_bar_c1 = (γ_k * sum(y_hat_θ_c1) / len(y_hat_θ)).to(device)  # u^bar_k(c)
        u_bar_c0_list.append(u_bar_c0)
        u_bar_c1_list.append(u_bar_c1)

    j_hat_c0_p0 = torch.tensor(j_bar_c0_p0_list).sum()
    j_hat_c0_p1 = torch.tensor(j_bar_c0_p1_list).sum()
    j_hat_c1_p0 = torch.tensor(j_bar_c1_p0_list).sum()
    j_hat_c1_p1 = torch.tensor(j_bar_c1_p1_list).sum()
    u_hat_c0 = torch.tensor(u_bar_c0_list).sum()
    u_hat_c1 = torch.tensor(u_bar_c1_list).sum()

    q_hat_00 = j_hat_c0_p0 * r_hat_p0 / torch.sqrt(u_hat_c0 * r_hat_p0)
    q_hat_01 = j_hat_c0_p1 * r_hat_p1 / torch.sqrt(u_hat_c0 * r_hat_p1)
    q_hat_10 = j_hat_c1_p0 * r_hat_p0 / torch.sqrt(u_hat_c1 * r_hat_p0)
    q_hat_11 = j_hat_c1_p1 * r_hat_p1 / torch.sqrt(u_hat_c1 * r_hat_p1)
    Q_hat = torch.tensor([
        [q_hat_00, q_hat_01],
        [q_hat_10, q_hat_11]
    ]).to(device)

    u, s, v = torch.svd(Q_hat)

    second_singular_vector_of_Q_hat = v[1].reshape(-1, 1).to(device)
    return second_singular_vector_of_Q_hat


def Fed_Renyi_LR(device,
                 mask_s1_flag,
                 lamda,
                 global_model,
                 algorithm_epoch_T, num_clients_K, communication_round_I, FL_fraction, FL_drop_rate, local_step_size,
                 training_dataloaders,
                 training_dataset,
                 client_dataset_list,
                 γ_k_style
                 ):
    training_dataset_size = len(training_dataset)
    client_datasets_size_list = [len(item) for item in client_dataset_list]

    # Training process
    logger.info("Training process")

    # Parameter Initialization
    global_model.train()
    local_model_list = [copy.deepcopy(global_model) for i in range(num_clients_K)]

    # global_v_0 = global_v_0_initialization(num_clients_K, mask_s1_flag, training_dataset, client_dataset_list)
    global_v = torch.rand(2, 1)

    r_hat_p0 = r_hat_p_initialization(num_clients_K, mask_s1_flag, training_dataset, client_dataset_list, p=0)
    r_hat_p1 = r_hat_p_initialization(num_clients_K, mask_s1_flag, training_dataset, client_dataset_list, p=1)

    criterion = torch.nn.BCELoss()

    for iter_t in range(algorithm_epoch_T):
        # Simulate Client Parallel
        for i in range(num_clients_K):
            model = local_model_list[i]
            model.train()
            optimizer = torch.optim.SGD(model.parameters(), lr=local_step_size)
            client_i_dataloader = training_dataloaders[i]

            # local option
            logger.info(f"########## Algorithm Epoch: {iter_t + 1} / {algorithm_epoch_T}; "
                        f"Client: {i + 1} / {num_clients_K};  ##########")
            for batch_index, batch in enumerate(client_i_dataloader):
                X = batch["X"].to(device)
                y = batch["y"].reshape(-1, 1).to(device)
                if mask_s1_flag:
                    s = batch["s2"]
                else:
                    s = batch["s1"]

                local_prediction = model(X).to(device)
                loss = criterion(local_prediction, y.float())
                y_hat_θ = (local_prediction >= 0.5).reshape(-1).to(device)
                regularization_term = lamda * G_θ_v(y_hat_θ, s, global_v).to(device)
                regularization_term = torch.where(torch.isnan(regularization_term), torch.full_like(regularization_term, 0), regularization_term)

                if torch.isnan(regularization_term):
                    logger.info("Regularization term is nan, now fix it to 0.")
                if batch_index % 10 == 0:
                    logger.info(f"      @@@@ "
                                f"Batch: {batch_index}; "
                                f"Cross Entropy Loss: {round(float(loss),4)}; "
                                f"Regularization term: {round(float(regularization_term),4)}; "
                                f"Total Loss: {round(float(loss+regularization_term),4)}; "
                                f"@@@@")

                loss += regularization_term
                loss.backward()
                optimizer.step()

            # Upgrade the local model list
            local_model_list[i] = model

        # Communicate
        if (iter_t + 1) % communication_round_I == 0:
            logger.info(f"********** Communicate: {(iter_t + 1) / communication_round_I} **********")
            # Client selection
            idxs_users = client_selection(
                client_num=num_clients_K,
                fraction=FL_fraction,
                dataset_size=training_dataset_size,
                client_dataset_size_list=client_datasets_size_list,
                drop_rate=FL_drop_rate,
                style="FedAvg",
            )
            logger.info(f"Select client list: {idxs_users} ")

            # Global operation
            theta_list = []
            for id in idxs_users:
                selected_model = local_model_list[id]
                if "uniform_client" in γ_k_style:
                    γ_k = 1 / len(idxs_users)
                else:
                    γ_k = len([training_dataset[idx]['X'] for idx in client_dataset_list[id].indices]) / training_dataset_size

                theta_list.append(list(γ_k * np.array(get_parameters(selected_model))))

            logger.info("********** Parameter aggregation **********")
            theta_list = np.array(theta_list, dtype=object)
            theta_avg = np.sum(theta_list, 0).tolist()
            set_parameters(global_model, theta_avg)

            logger.info("********** Global v update **********")
            backup_v = global_v

            try:
                global_v = argmax_v_LR(idxs_users, local_model_list, mask_s1_flag, training_dataset, client_dataset_list, r_hat_p0, r_hat_p1, γ_k_style)
            except Exception:
                global_v = backup_v
            # Parameter Distribution
            logger.info("********** Parameter distribution **********")
            local_model_list = [copy.deepcopy(global_model) for i in range(num_clients_K)]

    logger.info("Training finish, return global model")
    return global_model


def Fed_Renyi_NN(device,
                 mask_s1_flag,
                 lamda,
                 global_model,
                 algorithm_epoch_T, num_clients_K, communication_round_I, FL_fraction, FL_drop_rate, local_step_size,
                 training_dataloaders,
                 training_dataset,
                 client_dataset_list,
                 γ_k_style
                 ):
    training_dataset_size = len(training_dataset)
    client_datasets_size_list = [len(item) for item in client_dataset_list]

    # Training process
    logger.info("Training process")

    # Parameter Initialization
    global_model.train()
    local_model_list = [copy.deepcopy(global_model) for i in range(num_clients_K)]

    # global_v_0 = global_v_0_initialization(num_clients_K, mask_s1_flag, training_dataset, client_dataset_list)
    global_v = torch.rand(2, 1)

    r_hat_p0 = r_hat_p_initialization(num_clients_K, mask_s1_flag, training_dataset, client_dataset_list, p=0)
    r_hat_p1 = r_hat_p_initialization(num_clients_K, mask_s1_flag, training_dataset, client_dataset_list, p=1)

    criterion = torch.nn.CrossEntropyLoss()

    for iter_t in range(algorithm_epoch_T):
        # Simulate Client Parallel
        for i in range(num_clients_K):
            model = local_model_list[i]
            model.train()
            optimizer = torch.optim.SGD(model.parameters(), lr=local_step_size)
            client_i_dataloader = training_dataloaders[i]

            # local option
            logger.info(f"########## Algorithm Epoch: {iter_t + 1} / {algorithm_epoch_T}; "
                        f"Client: {i + 1} / {num_clients_K};  ##########")
            for batch_index, batch in enumerate(client_i_dataloader):
                X = batch["X"].to(device)
                y = batch["y"].to(device)
                if mask_s1_flag:
                    s = batch["s2"]
                else:
                    s = batch["s1"]

                local_prediction = model(X).to(device)
                loss = criterion(local_prediction, y.long())
                y_hat_θ = torch.argmax(local_prediction, dim=1)
                regularization_term = lamda * G_θ_v(y_hat_θ, s, global_v)
                regularization_term = torch.where(torch.isnan(regularization_term), torch.full_like(regularization_term, 0), regularization_term)

                if torch.isnan(regularization_term):
                    logger.info("Regularization term is nan, now fix it to 0.")
                if batch_index % 10 == 0:
                    logger.info(f"      @@@@ "
                                f"Batch: {batch_index}; "
                                f"Cross Entropy Loss: {round(float(loss),4)}; "
                                f"Regularization term: {round(float(regularization_term),4)}; "
                                f"Total Loss: {round(float(loss+regularization_term),4)}; "
                                f"@@@@")

                loss += regularization_term
                loss.backward()
                optimizer.step()

            # Upgrade the local model list
            local_model_list[i] = model

        # Communicate
        if (iter_t + 1) % communication_round_I == 0:
            logger.info(f"********** Communicate: {(iter_t + 1) / communication_round_I} **********")
            # Client selection
            idxs_users = client_selection(
                client_num=num_clients_K,
                fraction=FL_fraction,
                dataset_size=training_dataset_size,
                client_dataset_size_list=client_datasets_size_list,
                drop_rate=FL_drop_rate,
                style="FedAvg",
            )
            logger.info(f"Select client list: {idxs_users} ")

            # Global operation
            theta_list = []
            for id in idxs_users:
                selected_model = local_model_list[id]
                if "uniform_client" in γ_k_style:
                    γ_k = 1 / len(idxs_users)
                else:
                    γ_k = len([training_dataset[idx]['X'] for idx in client_dataset_list[id].indices]) / training_dataset_size

                theta_list.append(list(γ_k * np.array(get_parameters(selected_model))))

            logger.info("********** Parameter aggregation **********")
            theta_list = np.array(theta_list, dtype=object)
            theta_avg = np.sum(theta_list, 0).tolist()
            set_parameters(global_model, theta_avg)

            logger.info("********** Global v update **********")
            global_v = argmax_v_NN(idxs_users, local_model_list, mask_s1_flag, training_dataset, client_dataset_list, r_hat_p0, r_hat_p1, γ_k_style)

            # Parameter Distribution
            logger.info("********** Parameter distribution **********")
            local_model_list = [copy.deepcopy(global_model) for i in range(num_clients_K)]

    logger.info("Training finish, return global model")
    return global_model
