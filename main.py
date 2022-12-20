import os
import torch
import datetime
import numpy as np

import pickle

from hypothesis.LogisticRegression import RenyiLogisticRegression
from hypothesis.NeuralNetwork import RenyiNeuralNetwork
from data.dataset import get_ADULT_dataset, get_COMPAS_dataset, get_DRUG_dataset
from data.dataloader import get_FL_dataloader
from tool.utils import get_specific_time, get_parameters, set_parameters
from tool.logger import *
from algorithm.FederatedRenyi import Fed_Renyi
from algorithm.FederatedAverage import Fed_AVG_NN, Fed_AVG_LR


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
        logger.info("Model Construction (Logistic Regression)")
        model = RenyiLogisticRegression(input_size=nn_input_size)
    else:
        logger.info("Model Construction (Neural Network)")
        model = RenyiNeuralNetwork(input_size=nn_input_size, hidden_size=param_dict['nn_hidden_size'])
    return model


def Experiment_Federated_Average(param_dict, global_model, training_dataloaders, training_dataset, client_dataset_list, testing_dataloader):
    acc_numerator = 0
    acc_denominator = 0
    if param_dict['hypothesis'] == "LR":
        updated_global_model = Fed_AVG_LR(
            global_model,
            param_dict['algorithm_epoch_T'],
            param_dict['num_clients_m'],
            param_dict['local_update_round_K'],
            param_dict['FL_fraction'],
            param_dict['FL_drop_rate'],
            param_dict['local_step_size'],
            training_dataloaders,
            training_dataset,
            client_dataset_list
        )
        # Model testing
        logger.info("Global model testing")
        for batch_index, batch in enumerate(testing_dataloader):
            X = batch["X"]
            y = batch["y"].reshape(-1, 1)
            global_prediction = global_model(X).reshape(-1, 1)
            acc_numerator += int(sum((global_prediction >= 0.5).eq(y)))
            acc_denominator += X.shape[0]

    else:
        updated_global_model = Fed_AVG_NN(
            global_model,
            param_dict['algorithm_epoch_T'],
            param_dict['num_clients_m'],
            param_dict['local_update_round_K'],
            param_dict['FL_fraction'],
            param_dict['FL_drop_rate'],
            param_dict['local_step_size'],
            training_dataloaders,
            training_dataset,
            client_dataset_list
        )
        # Model testing
        logger.info("Global model testing")
        for batch_index, batch in enumerate(testing_dataloader):
            X = batch["X"]
            y = batch["y"]
            global_prediction = torch.argmax(global_model(X), dim=1)
            acc_numerator += int(sum(global_prediction.eq(y)))
            acc_denominator += X.shape[0]

    acc = acc_numerator / acc_denominator
    logger.info(f"Global model acc: {acc}")


def Experiment(param_dict):
    # Create dataset
    logger.info("Creating dataset")
    training_dataset, testing_dataset, nn_input_size = Experiment_Create_dataset(param_dict)

    # Create dataloader
    logger.info("Creating dataloader")
    training_dataloaders, validation_dataloaders, client_dataset_list, testing_dataloader = Experiment_Create_dataloader(param_dict, training_dataset, testing_dataset)

    # Model Construction
    logger.info("Model construction")
    global_model = Experiment_Model_construction(param_dict, nn_input_size)

    # Federated Average
    logger.info("Federated Average")
    Experiment_Federated_Average(
        param_dict, global_model, training_dataloaders, training_dataset, client_dataset_list, testing_dataloader
    )


    # Federated Renyi
    # logger.info("Federated Renyi")
    # Fed_Renyi(global_model, local_model_list,
    #           algorithm_epoch_T, num_clients_m, local_update_round_K, FL_fraction, FL_drop_rate, local_step_size,
    #           ADULT_training_dataloaders,
    #           ADULT_training_dataset,
    #           ADULT_client_dataset_list
    #           )


def main():
    ################################################################################################
    # Create the log
    LOG_PATH = "./log_path"
    datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    nowTime = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    log_path = os.path.join(LOG_PATH, "train" + nowTime)
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    ################################################################################################
    # Hyper-params
    param_dict = {}
    param_dict['device'] = "cuda" if torch.cuda.is_available() else "cpu"  # Get cpu or gpu device for experiment
    param_dict['hypothesis'] = "NN"
    # param_dict['dataset_name'] = "ADULT"
    # param_dict['dataset_name'] = "COMPAS"
    param_dict['dataset_name'] = "DRUG"
    param_dict['need_validation'] = False  # Whether to make the validation process
    param_dict['mask_s1_flag'] = False  # Whether to mask the sensitive attribute s1
    param_dict['mask_s2_flag'] = False  # Whether to mask the sensitive attribute s1
    param_dict['mask_s1_s2_flag'] = False  # Whether to mask the sensitive attribute s1 & s2
    param_dict['FL_fraction'] = 0.2
    param_dict['FL_drop_rate'] = 0
    param_dict['local_step_size'] = 0.01  # Copy From Renyi
    param_dict['global_step_size'] = 0.01
    if param_dict['hypothesis'] == "LR":
        # param_dict['batch_size'] = 64
        param_dict['batch_size'] = 8
        # num_iterations'] = 5000  # Copy From Renyi
        param_dict['local_update_round_K'] = 3
    else:
        param_dict['local_update_round_K'] = 3
        # param_dict['local_update_round_K'] = 50  # Copy From Renyi
        param_dict['batch_size'] = 8
        # param_dict['batch_size'] = 128  # Copy From Renyi
        param_dict['nn_hidden_size'] = 12  # Copy from FedFair

    param_dict['num_clients_m'] = 10
    # param_dict['num_clients_m'] = 50  # Copy from FedFair, ADULT
    # param_dict['num_clients_m'] = 20  # FedFair, COMPAS
    # param_dict['num_clients_m'] = 10  # FedFair, DRUG



    param_dict['algorithm_epoch_T'] = 2  # Customized
    ################################################################################################
    # Parameter announcement
    logger.info("Parameter announcement")
    for para_key in list(param_dict.keys()):
        logger.info(f"****** {para_key} : {param_dict[para_key]} ****** device: ")
    ################################################################################################
    Experiment(param_dict)
    # # Capture the exception
    # try:
    #     Experiment(param_dict)
    # except KeyboardInterrupt:
    #     logger.error("[Error] Caught keyboard interrupt on worker.")
    #     # save_model(model, os.path.join(train_dir, "earlystop"))
    # except Exception as e:
    #     logger.error("[Error] Other Error. ")
    #     logger.error(f"[Error] Error info: {e}")
    #     # save_model(model, os.path.join(train_dir, "OtherError"))


if __name__ == '__main__':
    main()
