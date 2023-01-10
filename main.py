import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import torch
import datetime
import numpy as np
import json
import pickle
import statistics

from hypothesis.LogisticRegression import RenyiLogisticRegression
from hypothesis.NeuralNetwork import RenyiNeuralNetwork
from moudle.dataset import get_ADULT_dataset, get_COMPAS_dataset, get_DRUG_dataset
from moudle.dataloader import get_FL_dataloader
from tool.logger import *
from algorithm.FederatedRenyi import Fed_Renyi_NN, Fed_Renyi_LR
from algorithm.FederatedAverage import Fed_AVG_NN, Fed_AVG_LR
from algorithm.FederatedFair import Fed_Fair_NN, Fed_Fair_LR


def Experiment_Create_dataset(param_dict, no_pickle=False):
    dataset_name = param_dict['dataset_name']
    mask_s1_flag = param_dict['mask_s1_flag']
    mask_s2_flag = param_dict['mask_s2_flag']
    mask_s1_s2_flag = param_dict['mask_s1_s2_flag']

    if "ADULT" in dataset_name:
        pickle_path = "./dataset/ADULT/ADULT.pickle"
        data_path = "./dataset/ADULT"
        get_dataset = get_ADULT_dataset
    elif "COMPAS" in dataset_name:
        pickle_path = "./dataset/COMPAS/COMPAS.pickle"
        data_path = "./dataset/COMPAS"
        get_dataset = get_COMPAS_dataset
    else:
        pickle_path = "./dataset/DRUG/DRUG.pickle"
        data_path = "./dataset/DRUG"
        get_dataset = get_DRUG_dataset

    if not os.path.exists(pickle_path) or no_pickle:
        training_dataset, testing_dataset = get_dataset(data_path, mask_s1_flag, mask_s2_flag, mask_s1_s2_flag)
        pickle_dict = {
            "training_dataset": training_dataset,
            "testing_dataset": testing_dataset,
        }
        with open(pickle_path, 'wb') as p:
            pickle.dump(pickle_dict, p)
            p.close()
        logger.info(f"Data Info (Training): {len(training_dataset)}")
        logger.info(f"Data Info (Training): {len(testing_dataset)}")
    else:
        with open(pickle_path, 'rb') as r:
            pickle_dict = pickle.load(r)
            r.close()
        training_dataset = pickle_dict['training_dataset']
        testing_dataset = pickle_dict['testing_dataset']
        logger.info(f"Data Info (Training): {len(training_dataset)}")
        logger.info(f"Data Info (Training): {len(testing_dataset)}")

    # Whether to mask the sensitive attribute
    nn_input_size = training_dataset.X.shape[1]

    if mask_s1_flag:
        logger.info("Masking the sensitive attribute s1")
    elif mask_s2_flag:
        logger.info("Masking the sensitive attribute s2")
    elif mask_s1_s2_flag:
        logger.info("Masking the sensitive attribute s1 and s2")
    else:
        logger.info("Do not masking the sensitive attribute")

    return training_dataset, testing_dataset, nn_input_size


def Experiment_Create_dataloader(param_dict, training_dataset, testing_dataset):
    num_clients_m = param_dict['num_clients_m']
    batch_size = param_dict['batch_size']
    need_validation = param_dict['need_validation']

    training_dataloaders, validation_dataloaders, client_dataset_list = get_FL_dataloader(
        training_dataset, num_clients_m, split_strategy="Uniform",
        do_train=True, need_validation=need_validation, batch_size=batch_size,
        num_workers=0, do_shuffle=True
    )

    testing_dataloader = get_FL_dataloader(
        testing_dataset, num_clients_m, split_strategy="Uniform",
        do_train=False, batch_size=batch_size, num_workers=0
    )
    return training_dataloaders, validation_dataloaders, client_dataset_list, testing_dataloader


def Experiment_Model_construction(param_dict, nn_input_size):
    if param_dict['hypothesis'] == "LR":
        logger.info("Model construction (Logistic Regression)")
        model = RenyiLogisticRegression(input_size=nn_input_size)
    else:
        logger.info("Model construction (Neural Network)")
        hidden_size = 12  # Copy from FedFair
        model = RenyiNeuralNetwork(input_size=nn_input_size, hidden_size=hidden_size)

    device = param_dict['device']
    model.to(device)
    return model


def Experiment_Federated_Average(param_dict, global_model, training_dataloaders, training_dataset, client_dataset_list,
                                 testing_dataloader):
    device = param_dict['device']

    if param_dict['hypothesis'] == "LR":
        updated_global_model = Fed_AVG_LR(
            device,
            global_model,
            param_dict['algorithm_epoch_T'],
            param_dict['num_clients_m'],
            param_dict['communication_round_I'],
            param_dict['FL_fraction'],
            param_dict['FL_drop_rate'],
            param_dict['local_step_size'],
            training_dataloaders,
            training_dataset,
            client_dataset_list
        )
        logger.info("-----------------------------------------------------------------------------")

    else:
        updated_global_model = Fed_AVG_NN(
            device,
            global_model,
            param_dict['algorithm_epoch_T'],
            param_dict['num_clients_m'],
            param_dict['communication_round_I'],
            param_dict['FL_fraction'],
            param_dict['FL_drop_rate'],
            param_dict['local_step_size'],
            training_dataloaders,
            training_dataset,
            client_dataset_list
        )
        logger.info("-----------------------------------------------------------------------------")

    Experiment_Model_testing(device, testing_dataloader, param_dict['mask_s1_flag'], updated_global_model,
                             param_dict['hypothesis'])


def Experiment_Federated_Renyi(param_dict, global_model, training_dataloaders, training_dataset, client_dataset_list,
                               testing_dataloader):
    device = param_dict['device']

    if param_dict['hypothesis'] == "LR":
        updated_global_model = Fed_Renyi_LR(
            device,
            param_dict['mask_s1_flag'],
            param_dict['lamda'],
            global_model,
            param_dict['algorithm_epoch_T'],
            param_dict['num_clients_m'],
            param_dict['communication_round_I'],
            param_dict['FL_fraction'],
            param_dict['FL_drop_rate'],
            param_dict['local_step_size'],
            training_dataloaders,
            training_dataset,
            client_dataset_list
        )
        logger.info("-----------------------------------------------------------------------------")

    else:
        updated_global_model = Fed_Renyi_NN(
            device,
            param_dict['mask_s1_flag'],
            param_dict['lamda'],
            global_model,
            param_dict['algorithm_epoch_T'],
            param_dict['num_clients_m'],
            param_dict['communication_round_I'],
            param_dict['FL_fraction'],
            param_dict['FL_drop_rate'],
            param_dict['local_step_size'],
            training_dataloaders,
            training_dataset,
            client_dataset_list
        )
        logger.info("-----------------------------------------------------------------------------")

    # Model testing
    Experiment_Model_testing(device, testing_dataloader, param_dict['mask_s1_flag'], updated_global_model,
                             param_dict['hypothesis'])


def Experiment_Federated_Fair(param_dict, global_model, training_dataloaders, training_dataset, client_dataset_list,
                              testing_dataloader):
    device = param_dict['device']
    hypothesis = param_dict['hypothesis']

    if "ADULT" in param_dict['dataset_name']:
        ϵ = 0.1
    elif "COMPAS" in param_dict['dataset_name']:
        if "LR" in param_dict['hypothesis']:
            ϵ = 0.0001
        else:
            ϵ = 0.01
    else:
        if "LR" in param_dict['hypothesis']:
            ϵ = 0.07
        else:
            ϵ = 0.005
    if param_dict['hypothesis'] == "LR":
        updated_global_model = Fed_Fair_LR(
            device,
            global_model,
            param_dict['algorithm_epoch_T'], param_dict['num_clients_m'],
            training_dataloaders,
            training_dataset,
            client_dataset_list,
            ϵ
        )
        logger.info("-----------------------------------------------------------------------------")

    else:
        updated_global_model = Fed_Fair_NN(
            device,
            global_model,
            param_dict['algorithm_epoch_T'], param_dict['num_clients_m'],
            training_dataloaders,
            training_dataset,
            client_dataset_list,
            ϵ
        )
        logger.info("-----------------------------------------------------------------------------")

    # Model testing
    Experiment_Model_testing(device, testing_dataloader, param_dict['mask_s1_flag'], updated_global_model, hypothesis)


def Experiment_Model_testing(device, testing_dataloader, mask_s1_flag, global_model, hypothesis):

    acc_numerator = 0
    acc_denominator = 0

    num_s1_pred1 = 0
    num_s1_pred0 = 0
    num_s0_pred1 = 0
    num_s0_pred0 = 0

    # Model testing
    logger.info("Global model testing")
    for batch_index, batch in enumerate(testing_dataloader):
        X = batch["X"].to(device)
        y = batch["y"].to(device)
        if hypothesis == "LR":
            global_prediction = (global_model(X) >= 0.5).to(device).reshape(-1)
        else:
            global_prediction = global_model(X).to(device).argmax(dim=1)
        acc_numerator += int(sum(global_prediction.eq(y)))
        acc_denominator += X.shape[0]

        if mask_s1_flag:
            s = batch["s2"]
        else:
            s = batch["s1"]

        y_1 = (y == 1).int().reshape(-1).to(device)
        s_1 = (s == 1).int().to(device)
        s_0 = (s == 0).int().to(device)
        pred_1 = (global_prediction == 1).int().to(device)
        pred_0 = (global_prediction == 0).int().to(device)

        num_s1_pred1 += (y_1 * s_1 * pred_1).sum().to(device)
        num_s1_pred0 += (y_1 * s_1 * pred_0).sum().to(device)
        num_s0_pred1 += (y_1 * s_0 * pred_1).sum().to(device)
        num_s0_pred0 += (y_1 * s_0 * pred_0).sum().to(device)

        # logger.info(f"num_s1_pred1: {num_s1_pred1} ")
        # logger.info(f"num_s1_pred0: {num_s1_pred0} ")
        # logger.info(f"num_s0_pred1: {num_s0_pred1} ")
        # logger.info(f"num_s0_pred0: {num_s0_pred0} ")

    acc = acc_numerator / acc_denominator
    logger.info(f"Global model acc: {acc}")
    x1 = num_s1_pred1 / (num_s1_pred1 + num_s1_pred0)
    logger.info(f"P(y = 1 | s = 1) = {x1} ")
    x2 = num_s0_pred1 / (num_s0_pred1 + num_s0_pred0)
    logger.info(f"P(y = 1 | s = 0) = {x2} ")
    # logger.info(f"DI: {x2 / x1} ")
    DEO = max(x2 - x1, x1 - x2)
    logger.info(f"Difference of Equality of Opportunity violation (DEO): {DEO}")
    FR = 1 - DEO
    logger.info(f"Fairness measurement (FR): {FR}")
    HM = statistics.harmonic_mean([acc, float(FR)])
    logger.info(f"Harmonic Mean of Fairness and Accuracy (HM): {HM}")
    
    


def Experiment(param_dict):
    # Create dataset
    logger.info("Creating dataset")
    training_dataset, testing_dataset, nn_input_size = Experiment_Create_dataset(param_dict)

    # Create dataloader
    logger.info("Creating dataloader")
    training_dataloaders, validation_dataloaders, client_dataset_list, testing_dataloader = Experiment_Create_dataloader(
        param_dict, training_dataset, testing_dataset)

    # Model Construction
    global_model = Experiment_Model_construction(param_dict, nn_input_size)
    logger.info("-----------------------------------------------------------------------------")

    if "FederatedAverage" in param_dict["algorithm"]:
        # Federated Average
        logger.info("~~~~~~ Algorithm: Federated Average ~~~~~~")
        Experiment_Federated_Average(
            param_dict, global_model, training_dataloaders, training_dataset, client_dataset_list, testing_dataloader
        )

        logger.info("#￥#￥#￥#￥#￥#￥#￥#￥#￥#￥#￥#￥#￥#￥#￥#￥#￥#￥#￥#￥#￥#￥#￥#￥#￥#￥#￥#￥#￥#￥#￥#￥#￥#￥#￥#￥")
        logger.info("#￥#￥#￥#￥#￥#￥#￥#￥#￥#￥#￥#￥#￥#￥#￥#￥#￥#￥#￥#￥#￥#￥#￥#￥#￥#￥#￥#￥#￥#￥#￥#￥#￥#￥#￥#￥")
        logger.info("#￥#￥#￥#￥#￥#￥#￥#￥#￥#￥#￥#￥#￥#￥#￥#￥#￥#￥#￥#￥#￥#￥#￥#￥#￥#￥#￥#￥#￥#￥#￥#￥#￥#￥#￥#￥")

    if "FederatedRenyi" in param_dict["algorithm"]:
        # Federated Renyi
        logger.info("~~~~~~ Algorithm: Federated Renyi ~~~~~~")
        Experiment_Federated_Renyi(
            param_dict, global_model, training_dataloaders, training_dataset, client_dataset_list, testing_dataloader
        )

        logger.info("#￥#￥#￥#￥#￥#￥#￥#￥#￥#￥#￥#￥#￥#￥#￥#￥#￥#￥#￥#￥#￥#￥#￥#￥#￥#￥#￥#￥#￥#￥#￥#￥#￥#￥#￥#￥")
        logger.info("#￥#￥#￥#￥#￥#￥#￥#￥#￥#￥#￥#￥#￥#￥#￥#￥#￥#￥#￥#￥#￥#￥#￥#￥#￥#￥#￥#￥#￥#￥#￥#￥#￥#￥#￥#￥")
        logger.info("#￥#￥#￥#￥#￥#￥#￥#￥#￥#￥#￥#￥#￥#￥#￥#￥#￥#￥#￥#￥#￥#￥#￥#￥#￥#￥#￥#￥#￥#￥#￥#￥#￥#￥#￥#￥")

    if "FederatedFair" in param_dict["algorithm"]:
        # Federated Fair
        logger.info("~~~~~~ Algorithm: Federated Fair ~~~~~~")
        Experiment_Federated_Fair(
            param_dict, global_model, training_dataloaders, training_dataset, client_dataset_list, testing_dataloader
        )

        logger.info("#￥#￥#￥#￥#￥#￥#￥#￥#￥#￥#￥#￥#￥#￥#￥#￥#￥#￥#￥#￥#￥#￥#￥#￥#￥#￥#￥#￥#￥#￥#￥#￥#￥#￥#￥#￥")
        logger.info("#￥#￥#￥#￥#￥#￥#￥#￥#￥#￥#￥#￥#￥#￥#￥#￥#￥#￥#￥#￥#￥#￥#￥#￥#￥#￥#￥#￥#￥#￥#￥#￥#￥#￥#￥#￥")
        logger.info("#￥#￥#￥#￥#￥#￥#￥#￥#￥#￥#￥#￥#￥#￥#￥#￥#￥#￥#￥#￥#￥#￥#￥#￥#￥#￥#￥#￥#￥#￥#￥#￥#￥#￥#￥#￥")

def main(dataset_name, hypothesis, algorithm):
    ################################################################################################
    # Hyper-params
    param_dict = {}
    with open("./COMMON.json", "r") as f:
        temp_dict = json.load(f)
    param_dict.update(**temp_dict)
    with open("./" + dataset_name + ".json", "r") as f:
        temp_dict = json.load(f)
    param_dict.update(**temp_dict)
    #param_dict['device'] = "cuda" if torch.cuda.is_available() else "cpu"  # Get cpu or gpu device for experiment
    param_dict['device'] = "cpu"
    Experiment_NO = 1
    lamda_list = [500, 100, 75, 50, 25, 10, 1, 0.01, 0]
    FL_drop_rate_list = [0, 0.1, 0.2, 0.3, 0.4, 0.5]
    algorithm_epoch_T_communication_round_I_list = [(10, 3), (30, 10), (50, 15), (100, 20)]
    # hypothesis_list = ["NN", "LR"]
    # algorithm_list = ["FederatedRenyi", "FederatedAverage"]

    param_dict['dataset_name'] = dataset_name
    param_dict['algorithm'] = algorithm
    param_dict['hypothesis'] = hypothesis

    # Skipping the unnecessary loop
    if "FederatedRenyi" not in algorithm:
        lamda_list = [0]

    for lamda in lamda_list:
        param_dict['lamda'] = lamda
        for FL_drop_rate in FL_drop_rate_list:
            param_dict['FL_drop_rate'] = FL_drop_rate
            for algorithm_epoch_T, communication_round_I in algorithm_epoch_T_communication_round_I_list:
                param_dict['algorithm_epoch_T'] = algorithm_epoch_T
                param_dict['communication_round_I'] = communication_round_I
                ################################################################################################
                # Create the log
                LOG_PATH = "./log_path/" + param_dict['dataset_name'] + "/" + param_dict['algorithm'] + "/" \
                           + param_dict['hypothesis'] + "/"
                # nowTime = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
                # log_path = os.path.join(LOG_PATH, "train" + nowTime )
                log_path = os.path.join(LOG_PATH, str(Experiment_NO))
                file_handler = logging.FileHandler(log_path+".txt")
                file_handler.setFormatter(formatter)
                logger.addHandler(file_handler)
                ################################################################################################
                logger.info(
                    f"Experiment {Experiment_NO}/{len(lamda_list) * len(FL_drop_rate_list) * len(algorithm_epoch_T_communication_round_I_list)} setup finish")
                json_str = json.dumps(param_dict, indent=4)
                with open(log_path + "_Parameter.json", "w") as json_file:
                    json_file.write(json_str)
                # Parameter announcement
                logger.info("Parameter announcement")
                for para_key in list(param_dict.keys()):
                    if "_common" in para_key:
                        continue
                    logger.info(f"****** {para_key} : {param_dict[para_key]} ******")
                logger.info("-----------------------------------------------------------------------------")
                ################################################################################################
                Experiment(param_dict)
                Experiment_NO += 1
                logger.info("|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||")
                logger.info("|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||")
                logger.info("")
                logger.info("")
                logger.info("")


if __name__ == '__main__':
    # print(sys.argv[1])
    # print(sys.argv[2])
    # print(sys.argv[3])

    main(sys.argv[1], sys.argv[2], sys.argv[3])
    # main(sys.argv[1], sys.argv[2], "FederatedFair")
    # main("COMPAS", "LR", "FederatedFair")
