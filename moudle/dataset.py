import os
import numpy as np
import csv
import random

from sklearn.preprocessing import OneHotEncoder
from torch.utils.data import Dataset


class ADULTDataset(Dataset):
    def __init__(self, attribute_dict):
        # self.raw_X = attribute_dict['raw_X']
        # self.raw_X_mask_s1 = attribute_dict['raw_X_mask_s1']
        # self.raw_X_mask_s2 = attribute_dict['raw_X_mask_s2']
        # self.raw_X_mask_s1_s2 = attribute_dict['raw_X_mask_s1_s2']

        self.s1 = np.array(attribute_dict['s1'])
        self.s2 = np.array(attribute_dict['s2'])
        # self.X_mask_s1_s2 = np.array(attribute_dict['X_mask_s1_s2'])
        # self.X_mask_s1 = np.array(attribute_dict['X_mask_s1'])
        # self.X_mask_s2 = np.array(attribute_dict['X_mask_s2'])
        self.X = attribute_dict['X']
        self.y = np.array(attribute_dict['y'])

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return {"s1": self.s1[idx], "s2": self.s2[idx],
                # "X_mask_s1_s2": self.X_mask_s1_s2[idx],
                # "X_mask_s1": self.X_mask_s1[idx],
                # "X_mask_s2": self.X_mask_s2[idx],
                "X": self.X[idx],
                "y": self.y[idx]
                }


class COMPASDataset(Dataset):
    def __init__(self, attribute_dict):
        # self.raw_X = attribute_dict['raw_X']
        # self.raw_X_mask_s1 = attribute_dict['raw_X_mask_s1']
        # self.raw_X_mask_s2 = attribute_dict['raw_X_mask_s2']
        # self.raw_X_mask_s1_s2 = attribute_dict['raw_X_mask_s1_s2']

        self.s1 = np.array(attribute_dict['s1'])
        self.s2 = np.array(attribute_dict['s2'])
        # self.X_mask_s1_s2 = np.array(attribute_dict['X_mask_s1_s2'])
        # self.X_mask_s1 = np.array(attribute_dict['X_mask_s1'])
        # self.X_mask_s2 = np.array(attribute_dict['X_mask_s2'])
        self.X = attribute_dict['X']
        self.y = np.array(attribute_dict['y'])

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return {"s1": self.s1[idx], "s2": self.s2[idx],
                # "X_mask_s1_s2": self.X_mask_s1_s2[idx],
                # "X_mask_s1": self.X_mask_s1[idx],
                # "X_mask_s2": self.X_mask_s2[idx],
                "X": self.X[idx],
                "y": self.y[idx]
                }


class DRUGDataset(Dataset):
    def __init__(self, attribute_dict):
        # self.raw_X = attribute_dict['raw_X']
        # self.raw_X_mask_s1 = attribute_dict['raw_X_mask_s1']
        # self.raw_X_mask_s2 = attribute_dict['raw_X_mask_s2']
        # self.raw_X_mask_s1_s2 = attribute_dict['raw_X_mask_s1_s2']

        self.s1 = np.array(attribute_dict['s1'])
        self.s2 = np.array(attribute_dict['s2'])
        # self.X_mask_s1_s2 = np.array(attribute_dict['X_mask_s1_s2'])
        # self.X_mask_s1 = np.array(attribute_dict['X_mask_s1'])
        # self.X_mask_s2 = np.array(attribute_dict['X_mask_s2'])
        self.X = attribute_dict['X']
        self.y = np.array(attribute_dict['y'])

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return {"s1": self.s1[idx], "s2": self.s2[idx],
                # "X_mask_s1_s2": self.X_mask_s1_s2[idx],
                # "X_mask_s1": self.X_mask_s1[idx],
                # "X_mask_s2": self.X_mask_s2[idx],
                "X": self.X[idx],
                "y": self.y[idx]
                }


def get_ADULT_dataset(data_path, mask_s1_flag, mask_s2_flag, mask_s1_s2_flag):
    # Some codes are borrow from https://github.com/optimization-for-data-driven-science/Renyi-Fair-Inference
    enc = OneHotEncoder()

    # Preprocess (training dataset)
    raw_X, raw_X_mask_s1, raw_X_mask_s2, raw_X_mask_s1_s2 = [], [], [], []
    y = []  # (Training set)Income over 50K (T:1, F:0)
    s1 = []  # (Training set)Sensitive feature (Male:1, Femal:0)
    s2 = []  # (Training set)Sensitive feature (White:1, non-White:0)

    with open(os.path.join(data_path, 'adult.data')) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for i, row in enumerate(csv_reader):
            if i == 0:  # Skipping the row of feature name
                continue
            if row[9] == "Male":
                s1.append(1)
            else:
                s1.append(0)

            if row[8] == "White":
                s2.append(1)
            else:
                s2.append(0)

            if '>50K' in row[14]:
                y.append(1)
            else:
                y.append(0)

            row_copy = row[:14]
            row_mask_s1_copy = row[:9] + row[10:14]
            row_mask_s2_copy = row[:8] + row[9:14]
            row_mask_s1_s2_copy = row[:8] + row[10:14]

            raw_X.append(row_copy)
            raw_X_mask_s1.append(row_mask_s1_copy)
            raw_X_mask_s2.append(row_mask_s2_copy)
            raw_X_mask_s1_s2.append(row_mask_s1_s2_copy)

    # Preprocess (testing dataset)
    raw_testX, raw_testX_mask_s1, raw_testX_mask_s2, raw_testX_mask_s1_s2 = [], [], [], []
    testY = []  # (Testing set)Income over 50K (T:1, F:0)
    testS1 = []  # (Testing set)Sensitive feature (Male:1, Female:0)
    testS2 = []  # (Testing set)Sensitive feature (White:1, non-White:0)
    with open(os.path.join(data_path, 'adult.test')) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for i, row in enumerate(csv_reader):
            if i == 0:  # Skipping the row of feature name
                continue

            if row[9] == "Male":
                testS1.append(1)
            else:
                testS1.append(0)

            if row[8] == "White":
                testS2.append(1)
            else:
                testS2.append(0)

            if '>50K' in row[14]:
                testY.append(1)
            else:
                testY.append(0)

            row_copy = row[:14]
            row_mask_s1_copy = row[:9] + row[10:14]
            row_mask_s2_copy = row[:8] + row[9:14]
            row_mask_s1_s2_copy = row[:8] + row[10:14]

            raw_testX.append(row_copy)
            raw_testX_mask_s1.append(row_mask_s1_copy)
            raw_testX_mask_s2.append(row_mask_s2_copy)
            raw_testX_mask_s1_s2.append(row_mask_s1_s2_copy)

    # One-hot Encoding (training_dataset)
    enc.fit(raw_X_mask_s1_s2 + raw_testX_mask_s1_s2)
    if mask_s1_flag:
        X_mask_s1_s2 = np.float32(enc.transform(raw_X_mask_s1_s2).toarray())
        X_mask_s1 = np.float32(np.append(X_mask_s1_s2, np.array([s2]).transpose(), axis=1))
    elif mask_s2_flag:
        X_mask_s1_s2 = np.float32(enc.transform(raw_X_mask_s1_s2).toarray())
        X_mask_s2 = np.float32(np.append(X_mask_s1_s2, np.array([s1]).transpose(), axis=1))
    elif mask_s1_s2_flag:
        X_mask_s1_s2 = np.float32(enc.transform(raw_X_mask_s1_s2).toarray())
    else:
        X_mask_s1_s2 = np.float32(enc.transform(raw_X_mask_s1_s2).toarray())
        X = np.float32(np.append(X_mask_s1_s2, np.array([s1, s2]).transpose(), axis=1))

    # One-hot Encoding (testing)
    if mask_s1_flag:
        testX_mask_s1_s2 = np.float32(enc.transform(raw_testX_mask_s1_s2).toarray())
        testX_mask_s1 = np.float32(np.append(testX_mask_s1_s2, np.array([testS2]).transpose(), axis=1))
    elif mask_s2_flag:
        testX_mask_s1_s2 = np.float32(enc.transform(raw_testX_mask_s1_s2).toarray())
        testX_mask_s2 = np.float32(np.append(testX_mask_s1_s2, np.array([testS1]).transpose(), axis=1))
    elif mask_s1_s2_flag:
        testX_mask_s1_s2 = np.float32(enc.transform(raw_testX_mask_s1_s2).toarray())
    else:
        testX_mask_s1_s2 = np.float32(enc.transform(raw_testX_mask_s1_s2).toarray())
        # testX_mask_s2 = np.float32(np.append(testX_mask_s1_s2, np.array([testS1]).transpose(), axis=1))
        testX = np.float32(np.append(testX_mask_s1_s2, np.array([testS1, testS2]).transpose(), axis=1))

    # Constructing the training dataset
    training_attribute_dict = {
        # 'raw_X': np.array(raw_X), 'raw_X_mask_s1': np.array(raw_X_mask_s1),
        # 'raw_X_mask_s2': np.array(raw_X_mask_s2), 'raw_X_mask_s1_s2': np.array(raw_X_mask_s1_s2),
        's1': s1, 's2': s2, 'y': y
    }
    if mask_s1_flag:
        training_attribute_dict['X'] = X_mask_s1
    elif mask_s2_flag:
        training_attribute_dict['X'] = X_mask_s2
    elif mask_s1_s2_flag:
        training_attribute_dict['X'] = X_mask_s1_s2
    else:
        training_attribute_dict['X'] = X

    training_dataset = ADULTDataset(attribute_dict=training_attribute_dict)

    # Constructing the testing dataset
    testing_attribute_dict = {
        # 'raw_X': np.array(raw_testX), 'raw_X_mask_s1': np.array(raw_testX_mask_s1),
        # 'raw_X_mask_s2': np.array(raw_testX_mask_s2), 'raw_X_mask_s1_s2': np.array(raw_testX_mask_s1_s2),
        's1': testS1, 's2': testS2, 'y': testY
    }
    if mask_s1_flag:
        testing_attribute_dict['X'] = testX_mask_s1
    elif mask_s2_flag:
        testing_attribute_dict['X'] = testX_mask_s2
    elif mask_s1_s2_flag:
        testing_attribute_dict['X'] = testX_mask_s1_s2
    else:
        testing_attribute_dict['X'] = testX

    testing_dataset = ADULTDataset(attribute_dict=testing_attribute_dict)

    return training_dataset, testing_dataset


def get_COMPAS_dataset(data_path, mask_s1_flag=False, mask_s2_flag=False, mask_s1_s2_flag=False):
    # Some codes are borrow from https://github.com/propublica/compas-analysis/blob/master/Compas%20Analysis.ipynb
    enc = OneHotEncoder()
    with open(os.path.join(data_path, 'compas-scores-two-years.csv')) as csv_file:

        csv_reader = csv.reader(csv_file)
        raw_data = []

        # Filtering
        for i, row in enumerate(csv_reader):
            if i == 0:  # Skipping the row of feature name
                continue

            if row[15] != '' and row[24] != '' and row[22] != '' and row[40] != '':
                if 30 >= int(row[15]) >= -30:  # Filtering by `days_b_screening_arrest`
                    if int(row[24]) != -1:  # Filtering by `is_recid`
                        if row[22] != "0":  # Filtering by `c_charge_degree`
                            if row[40] != 'N/A':  # Filtering by `score_text`
                                if row[9] == "African-American" or row[9] == "Caucasian":  # Filtering by `race`
                                    raw_data.append(row)


        # Splitting
        random.shuffle(raw_data)
        training_set = raw_data[:4800]
        testing_set = raw_data[4800:]

        # Training set
        raw_X, raw_X_mask_s1, raw_X_mask_s2, raw_X_mask_s1_s2 = [], [], [], []
        y = []  # (Training set)Not a recidivist (is_recid=0 -> 1; is_recid=1 -> 0)
        s1 = []  # (Training set)Sensitive feature (African-American:1, Caucasian:0)
        s2 = []
        for i, row in enumerate(training_set):
            if row[9] == "African-American":  # African-American:1, Caucasian:0
                s1.append(1)
            else:
                s1.append(0)

            if row[5] == "Male":  # Male:1, Female:0
                s2.append(1)
            else:
                s2.append(0)

            if int(row[24]) == 0:  # Not a recidivist (is_recid=0 -> 1; is_recid=1 -> 0)
                y.append(1)
            else:
                y.append(0)

            row_copy = row[5:6] + row[8:9] + row[9:10] + row[10:11] + row[12:16] + row[22:23] + row[39:41] + row[48:49]  # Filtering out excess features
            row_mask_s1_copy = row[5:6] + row[8:9] + row[10:11] + row[12:16] + row[22:23] + row[39:41] + row[48:49]
            row_mask_s2_copy = row[8:9] + row[9:10] + row[10:11] + row[12:16] + row[22:23] + row[39:41] + row[48:49]
            row_mask_s1_s2_copy = row[8:9] + row[10:11] + row[12:16] + row[22:23] + row[39:41] + row[48:49]

            # row_copy = row[:24] + row[25:-1]  # Filtering the label and the feature 'two_year_recid' in last column
            # row_mask_s1_copy = row[:9] + row[10:24] + row[25:-1]
            # row_mask_s2_copy = row[:5] + row[6:24] + row[25:-1]
            # row_mask_s1_s2_copy = row[:5] + row[6:8] + row[10:24] + row[25:-1]

            raw_X.append(row_copy)
            raw_X_mask_s1.append(row_mask_s1_copy)
            raw_X_mask_s2.append(row_mask_s2_copy)
            raw_X_mask_s1_s2.append(row_mask_s1_s2_copy)

        # Testing
        raw_testX, raw_testX_mask_s1, raw_testX_mask_s2, raw_testX_mask_s1_s2 = [], [], [], []
        testY = []  # (Testing set)Not a recidivist (T:1, F:0)
        testS1 = []  # (Testing set)Sensitive feature (African-American:1, Caucasian:0)
        testS2 = []  # (Testing set)Sensitive feature (Male:1, Female:0)

        for i, row in enumerate(testing_set):
            if row[9] == "African-American":  # African-American:1, Caucasian:0
                testS1.append(1)
            else:
                testS1.append(0)

            if row[5] == "Male":  # Male:1, Female:0
                testS2.append(1)
            else:
                testS2.append(0)

            if int(row[24]) == 0:  # Not a recidivist (is_recid=0->T:1; is_recid=1->F:0)
                testY.append(1)
            else:
                testY.append(0)

            row_copy = row[5:6] + row[8:9] + row[9:10] + row[10:11] + row[12:16] + row[22:23] + row[39:41] + row[48:49]  # Filtering out excess features
            row_mask_s1_copy = row[5:6] + row[8:9] + row[10:11] + row[12:16] + row[22:23] + row[39:41] + row[48:49]
            row_mask_s2_copy = row[8:9] + row[9:10] + row[10:11] + row[12:16] + row[22:23] + row[39:41] + row[48:49]
            row_mask_s1_s2_copy = row[8:9] + row[10:11] + row[12:16] + row[22:23] + row[39:41] + row[48:49]

            # row_copy = row[:24] + row[25:-1]  # Filtering the label and the feature 'two_year_recid' in last column
            # row_mask_s1_copy = row[:9] + row[10:24] + row[25:-1]
            # row_mask_s2_copy = row[:5] + row[6:24] + row[25:-1]
            # row_mask_s1_s2_copy = row[:5] + row[6:8] + row[10:24] + row[25:-1]

            raw_testX.append(row_copy)
            raw_testX_mask_s1.append(row_mask_s1_copy)
            raw_testX_mask_s2.append(row_mask_s2_copy)
            raw_testX_mask_s1_s2.append(row_mask_s1_s2_copy)

    # One-hot Encoding (training_dataset)
    enc.fit(raw_X_mask_s1_s2 + raw_testX_mask_s1_s2)

    if mask_s1_flag:
        X_mask_s1_s2 = np.float32(enc.transform(raw_X_mask_s1_s2).toarray())
        X_mask_s1 = np.float32(np.append(X_mask_s1_s2, np.array([s2]).transpose(), axis=1))
    elif mask_s2_flag:
        X_mask_s1_s2 = np.float32(enc.transform(raw_X_mask_s1_s2).toarray())
        X_mask_s2 = np.float32(np.append(X_mask_s1_s2, np.array([s1]).transpose(), axis=1))
    elif mask_s1_s2_flag:
        X_mask_s1_s2 = np.float32(enc.transform(raw_X_mask_s1_s2).toarray())
    else:
        X_mask_s1_s2 = np.float32(enc.transform(raw_X_mask_s1_s2).toarray())
        X = np.float32(np.append(X_mask_s1_s2, np.array([s1, s2]).transpose(), axis=1))

    # One-hot Encoding (testing)
    if mask_s1_flag:
        testX_mask_s1_s2 = np.float32(enc.transform(raw_testX_mask_s1_s2).toarray())
        testX_mask_s1 = np.float32(np.append(testX_mask_s1_s2, np.array([testS2]).transpose(), axis=1))
    elif mask_s2_flag:
        testX_mask_s1_s2 = np.float32(enc.transform(raw_testX_mask_s1_s2).toarray())
        testX_mask_s2 = np.float32(np.append(testX_mask_s1_s2, np.array([testS1]).transpose(), axis=1))
    elif mask_s1_s2_flag:
        testX_mask_s1_s2 = np.float32(enc.transform(raw_testX_mask_s1_s2).toarray())
    else:
        testX_mask_s1_s2 = np.float32(enc.transform(raw_testX_mask_s1_s2).toarray())
        # testX_mask_s2 = np.float32(np.append(testX_mask_s1_s2, np.array([testS1]).transpose(), axis=1))
        testX = np.float32(np.append(testX_mask_s1_s2, np.array([testS1, testS2]).transpose(), axis=1))

    # Constructing the training dataset
    training_attribute_dict = {
        # 'raw_X': np.array(raw_X), 'raw_X_mask_s1': np.array(raw_X_mask_s1),
        # 'raw_X_mask_s2': np.array(raw_X_mask_s2), 'raw_X_mask_s1_s2': np.array(raw_X_mask_s1_s2),
        's1': s1, 's2': s2, 'y': y
    }
    if mask_s1_flag:
        training_attribute_dict['X'] = X_mask_s1
    elif mask_s2_flag:
        training_attribute_dict['X'] = X_mask_s2
    elif mask_s1_s2_flag:
        training_attribute_dict['X'] = X_mask_s1_s2
    else:
        training_attribute_dict['X'] = X

    training_dataset = COMPASDataset(attribute_dict=training_attribute_dict)

    # Constructing the testing dataset
    testing_attribute_dict = {
        # 'raw_X': np.array(raw_testX), 'raw_X_mask_s1': np.array(raw_testX_mask_s1),
        # 'raw_X_mask_s2': np.array(raw_testX_mask_s2), 'raw_X_mask_s1_s2': np.array(raw_testX_mask_s1_s2),
        's1': testS1, 's2': testS2, 'y': testY
    }
    if mask_s1_flag:
        testing_attribute_dict['X'] = testX_mask_s1
    elif mask_s2_flag:
        testing_attribute_dict['X'] = testX_mask_s2
    elif mask_s1_s2_flag:
        testing_attribute_dict['X'] = testX_mask_s1_s2
    else:
        testing_attribute_dict['X'] = testX

    testing_dataset = COMPASDataset(attribute_dict=testing_attribute_dict)

    return training_dataset, testing_dataset


def get_DRUG_dataset(data_path, mask_s1_flag=False, mask_s2_flag=False, mask_s1_s2_flag=False):
    enc = OneHotEncoder()
    with open(os.path.join(data_path, 'drug_consumption.data')) as csv_file:
        csv_reader = csv.reader(csv_file)
        raw_data = []

        # Pre_process: Filtering
        for i, row in enumerate(csv_reader):
            if i == 0:  # Skipping the row of feature name
                continue
            raw_data.append(row)

        # Splitting
        random.shuffle(raw_data)
        training_set = raw_data[:1600]
        testing_set = raw_data[1600:]

        # Training set
        raw_X, raw_X_mask_s1, raw_X_mask_s2, raw_X_mask_s1_s2 = [], [], [], []
        y = []  # (Training set)Not abuse volatile substance (Not abuse:1 ; Abuse:0)
        s1 = []  # (Training set)Sensitive feature (White:1, Non-white:0)
        s2 = []  # (Training set)Sensitive feature (Male:1, Female:0)
        for i, row in enumerate(training_set):
            if float(row[5]) == -0.31685:  # White:1, Non-white:0
                s1.append(1)
            else:
                s1.append(0)

            if float(row[2]) < 0:  # Male:1, Female:0
                s2.append(1)
            else:
                s2.append(0)

            if row[31] == 'CL0':  # Not abuse volatile substance (Not abuse:1 ; Abuse:0)
                y.append(1)
            else:
                y.append(0)

            row_copy = row[:31]  # Filtering the label in last column
            row_mask_s1_copy = row[:5] + row[6:31]
            row_mask_s2_copy = row[:2] + row[3:31]
            row_mask_s1_s2_copy = row[:2] + row[3:5] + row[6:31]

            raw_X.append(row_copy)
            raw_X_mask_s1.append(row_mask_s1_copy)
            raw_X_mask_s2.append(row_mask_s2_copy)
            raw_X_mask_s1_s2.append(row_mask_s1_s2_copy)

        # Testing
        raw_testX, raw_testX_mask_s1, raw_testX_mask_s2, raw_testX_mask_s1_s2 = [], [], [], []
        testY = []  # (Testing set)Not abuse volatile substance (Not abuse:1 ; Abuse:0)
        testS1 = []  # (Testing set)Sensitive feature (White:1, Non-white:0)
        testS2 = []  # (Testing set)Sensitive feature (Male:1, Female:0)

        for i, row in enumerate(testing_set):
            if float(row[5]) == -0.31685:  # White:1, Non-white:0
                testS1.append(1)
            else:
                testS1.append(0)

            if float(row[2]) < 0:  # Male:1, Female:0
                testS2.append(1)
            else:
                testS2.append(0)

            if row[31] == 'CL0':  # Not abuse volatile substance (Not abuse:1 ; Abuse:0)
                testY.append(1)
            else:
                testY.append(0)

            row_copy = row[:31]  # Filtering the label in last column
            row_mask_s1_copy = row[:5] + row[6:31]
            row_mask_s2_copy = row[:2] + row[3:31]
            row_mask_s1_s2_copy = row[:2] + row[3:5] + row[6:31]

            raw_testX.append(row_copy)
            raw_testX_mask_s1.append(row_mask_s1_copy)
            raw_testX_mask_s2.append(row_mask_s2_copy)
            raw_testX_mask_s1_s2.append(row_mask_s1_s2_copy)

    # One-hot Encoding (training_dataset)
    enc.fit(raw_X_mask_s1_s2 + raw_testX_mask_s1_s2)
    if mask_s1_flag:
        X_mask_s1_s2 = np.float32(enc.transform(raw_X_mask_s1_s2).toarray())
        X_mask_s1 = np.float32(np.append(X_mask_s1_s2, np.array([s2]).transpose(), axis=1))
    elif mask_s2_flag:
        X_mask_s1_s2 = np.float32(enc.transform(raw_X_mask_s1_s2).toarray())
        X_mask_s2 = np.float32(np.append(X_mask_s1_s2, np.array([s1]).transpose(), axis=1))
    elif mask_s1_s2_flag:
        X_mask_s1_s2 = np.float32(enc.transform(raw_X_mask_s1_s2).toarray())
    else:
        X_mask_s1_s2 = np.float32(enc.transform(raw_X_mask_s1_s2).toarray())
        X = np.float32(np.append(X_mask_s1_s2, np.array([s1, s2]).transpose(), axis=1))

    # One-hot Encoding (testing)
    if mask_s1_flag:
        testX_mask_s1_s2 = np.float32(enc.transform(raw_testX_mask_s1_s2).toarray())
        testX_mask_s1 = np.float32(np.append(testX_mask_s1_s2, np.array([testS2]).transpose(), axis=1))
    elif mask_s2_flag:
        testX_mask_s1_s2 = np.float32(enc.transform(raw_testX_mask_s1_s2).toarray())
        testX_mask_s2 = np.float32(np.append(testX_mask_s1_s2, np.array([testS1]).transpose(), axis=1))
    elif mask_s1_s2_flag:
        testX_mask_s1_s2 = np.float32(enc.transform(raw_testX_mask_s1_s2).toarray())
    else:
        testX_mask_s1_s2 = np.float32(enc.transform(raw_testX_mask_s1_s2).toarray())
        # testX_mask_s2 = np.float32(np.append(testX_mask_s1_s2, np.array([testS1]).transpose(), axis=1))
        testX = np.float32(np.append(testX_mask_s1_s2, np.array([testS1, testS2]).transpose(), axis=1))

    # Constructing the training dataset
    training_attribute_dict = {
        # 'raw_X': np.array(raw_X), 'raw_X_mask_s1': np.array(raw_X_mask_s1),
        # 'raw_X_mask_s2': np.array(raw_X_mask_s2), 'raw_X_mask_s1_s2': np.array(raw_X_mask_s1_s2),
        's1': s1, 's2': s2, 'y': y
    }
    if mask_s1_flag:
        training_attribute_dict['X'] = X_mask_s1
    elif mask_s2_flag:
        training_attribute_dict['X'] = X_mask_s2
    elif mask_s1_s2_flag:
        training_attribute_dict['X'] = X_mask_s1_s2
    else:
        training_attribute_dict['X'] = X

    training_dataset = DRUGDataset(attribute_dict=training_attribute_dict)

    # Constructing the testing dataset
    testing_attribute_dict = {
        # 'raw_X': np.array(raw_testX), 'raw_X_mask_s1': np.array(raw_testX_mask_s1),
        # 'raw_X_mask_s2': np.array(raw_testX_mask_s2), 'raw_X_mask_s1_s2': np.array(raw_testX_mask_s1_s2),
        's1': testS1, 's2': testS2, 'y': testY
    }
    if mask_s1_flag:
        testing_attribute_dict['X'] = testX_mask_s1
    elif mask_s2_flag:
        testing_attribute_dict['X'] = testX_mask_s2
    elif mask_s1_s2_flag:
        testing_attribute_dict['X'] = testX_mask_s1_s2
    else:
        testing_attribute_dict['X'] = testX

    testing_dataset = DRUGDataset(attribute_dict=testing_attribute_dict)

    return training_dataset, testing_dataset
