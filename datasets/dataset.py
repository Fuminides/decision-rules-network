import os
import pickle

import torch
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, KBinsDiscretizer, LabelEncoder, LabelBinarizer
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from .FeatureBinarizer import FeatureBinarizer

import pandas as pd 
from torch.utils.data import Dataset, DataLoader
import re

def read_keel_header(file_path):
        '''
        Reads the header of the provided dataset file in KEEL format.
        The function stores the dataset parameters in the "header" attribute and the
        position of the first line after the header in the "data_startline" attribute.
        :param file_path: Path to the dataset file.
        :type file_path: str
        '''
        attributes = {}
        header = {
            'relation': {},
            'inputs': {},
            'outputs': {}
        }
        with open(file_path, 'r') as f:
            for i, line in enumerate(f):
                if line.endswith('\n'):
                    line = line[:-1]
                parts = line.split(' ')
                # Remove possible blank spaces at the end of the line
                parts = list(filter(lambda x: x != '', parts))

                if parts[0] == '@relation':
                    header['relation'] = parts[1]
                elif parts[0] == '@attribute' or parts[0] == '@Attribute':
                    attr_name = parts[1]
                    # CATEGORICAL VALUES:
                    if '{' in attr_name:
                        # Categorical data
                        attr_name, values = attr_name.split('{')
                        attributes[attr_name] = {}
                        attributes[attr_name]['type'] = 'categorical'
                        values[:-1].strip()  # Remove all possible blank spaces
                        range_vals = values[:-1].split(',')
                        attributes[attr_name]['range'] = range_vals
                    elif len(parts) == 3 and '{' in parts[2]:
                        # If there's an space between the attribute name and its values:
                        attr_name = attr_name
                        values = parts[2][1:-1]
                        attributes[attr_name] = {}
                        attributes[attr_name]['type'] = 'categorical'
                        values = values.split(',')
                        range_vals = values
                        attributes[attr_name]['range'] = range_vals
                    elif len(parts) > 3 and '{' in parts[2]:
                        # If there's an space between the attribute name and its values:
                        attr_name = attr_name
                        values = parts[2:]
                        attributes[attr_name] = {}
                        attributes[attr_name]['type'] = 'categorical'
                        values[0] = values[0].strip('{')
                        values[-1] = values[-1].strip('}')
                        for i in range(len(values)):
                            values[i] = values[i].split(',')[0]
                        range_vals = values
                        attributes[attr_name]['range'] = range_vals
                    # REAL VALUES:
                    elif parts[2].startswith('real'):
                        attributes[attr_name] = {}
                        attributes[attr_name]['type'] = 'real'
                        if len(parts) == 5:
                            # Sometimes there is a blank space between real and the attributes, and others there's not
                            range_vals = [*re.findall('\d+\.\d+', parts[3]), *re.findall('\d+\.\d+', parts[4])]
                        else:
                            range_vals = re.findall('\d+\.\d+', parts[2])
                        if range_vals == []:
                            range_vals = [-np.inf, np.inf]
                        attributes[attr_name]['range'] = (float(range_vals[0]), float(range_vals[1]))
                    # INTEGER VALUES:
                    elif parts[2].startswith('integer'):
                        attributes[attr_name] = {}
                        attributes[attr_name]['type'] = 'integer'
                        if len(parts) == 5:
                            range_vals = [*re.findall(r'\d+', parts[3]), *re.findall('\d+', parts[4])]
                        elif len(parts) == 4:
                            range_vals = re.findall(r'\d+', parts[3])
                        else:
                            range_vals = re.findall(r'\d+', parts[2])
                        attributes[attr_name]['range'] = (int(range_vals[0]), int(range_vals[1]))
                elif parts[0] == '@inputs':
                    for input in parts[1:]:
                        if input.endswith(','):
                            input = input[:-1]
                        if input != '':
                            # Filter possible final blank spaces
                            header['inputs'][input] = attributes[input]
                elif parts[0] == '@outputs' or parts[0] == '@output':
                    for output in parts[1:]:
                        header['outputs'][output] = attributes[output]
                elif parts[0] == '@data':
                    return header, i

        return header, -1



class auxKeelDataset(Dataset):

    def __init__(self, X, y) -> None:
        self.X = X
        self.y = y
        self.num_classes = len(torch.unique(y))
        self.num_attributes = X.shape[1]

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def transform_to_pytorch_dataset(X, y, device):
    '''
    Transforms the input and output data to a Pytorch dataset.
    '''
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y.values, dtype=torch.long)

    X = X.to(device=device)
    y = y.to(device=device)

    return auxKeelDataset(X, y)


def read_data(file_path, class_names=None, random_state=33):
    '''
    Reads and returns the data of the required partition/s of the provided dataset file in KEEL format.

    :param file_path: Path to the dataset file.
    :type file_path: str
    :class_names: Names of the classes of the dataset. If None, the class names will be inferred from the dataset.
    :type class_names: list<str>
    :param train: If None, all samples in the dataset will be returned.
        If True, the first 80% examples are returned (train-val split). Otherwise, the last 20% examples are returned (test split).
    :type train: bool
    :param precomputed_partitions: If True, precomputed (randomly and stratified) files dataset_train.dat and dataset_test.dat will be used.
        Otherwise, partitions will be computed from the same file as described by param "train".
    :type precomputed_partitions: bool
    '''
    header, data_startline = read_keel_header(file_path)
    if class_names is None:
        # If class associations haven't been provided, generate them (to rename them to values from 0 to num_classes-1):
        # Create association between dataset classes and predicted classes:
        class_names_ = {}
        output_attr_name = next(iter(header['outputs']))
        for i, class_name in enumerate(header['outputs'][output_attr_name]['range']):
            class_names_[class_name] = i
            # Read samples from the dataset:
    else:
        class_names_ = class_names
    # Read the corresponding samples:
    data = pd.read_csv(file_path, sep=',', header=data_startline,
                        names=(*header['inputs'].keys(), *header['outputs'].keys()), skipinitialspace=True)
    # Remove possible blank spaces at the end of the line
    for column in data.columns:
        if data[column].dtype == 'object':  # Check if the column is of type object (string)
            data[column] = data[column].map(lambda x: x.strip() if isinstance(x, str) else x)

    data_input = data[[key for key in header['inputs'].keys()]]
    # One-hot encode categorical input variables:
    categorical_columns = [key for key in header['inputs'].keys() if header['inputs'][key]['type'] == 'categorical']
    boolean_cateogrical_columns = [key in categorical_columns for key in header['inputs'].keys() ]
    boolean_real_columns = np.logical_not(boolean_cateogrical_columns)

    # There may be categorical variables with values that are not present in the training set:
    # In order to ensure that pd.get_dummies creates a column for such non-existent values, we set the type of each categorical column
    # to pd.CategoricalDtype prior to calling pd.get_dummies, specifying all potential values (even non-present ones):
    # https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy)
    column_types = {}
    for column in categorical_columns:
        comun_data, unique_classes = pd.factorize(data_input[column])
        data_input = data_input.assign(**{column: comun_data})
    #data_input = data_input.astype(column_types)
    # data_input = pd.get_dummies(data_input, columns=categorical_columns)

    data_target = data[header['outputs'].keys()]
    # Rename classes to the ordered keys:
    output_name = list(header['outputs'].keys())[0]
    data_target = data_target[output_name].apply(lambda x: class_names_[str(x)])
    # Standardize all columns:
    # data_input = (data_input - data_input.mean()) / (data_input.std() + float_info.epsilon)
    data_input = data_input.copy()
    data_input.loc[:, boolean_real_columns] = data_input.loc[:, boolean_real_columns].astype(float)
    data_different_min = data_input.loc[:, boolean_real_columns].min()
    data_different_max = data_input.loc[:, boolean_real_columns].max()
    import warnings
    warnings.filterwarnings("ignore")
    data_input.loc[:, boolean_real_columns] = (data_input.loc[:, boolean_real_columns] - data_different_min) / (
        data_different_max - data_different_min)

    # Code the categorical variables as integers:
    '''from sklearn import preprocessing
    for column in categorical_columns:
        le = preprocessing.LabelEncoder()
        aux = le.fit_transform(data_input.loc[:, column])
        data_input.drop(column, axis=1, inplace=True)
        # Change the data type of the column to the data type of aux
        data_input[column] = aux'''


    return data_input, data_target, header, class_names_, boolean_cateogrical_columns

def load_KeelDatasetPytorch(file_path, batch_size, train_proportion=0.8, random_state=33):
    '''
    :param file_path: Path to the dataset file.
    :type file_path: str
    :param batch_size: Number of samples in each minibatch.
    :type batch_size: int
    :param train: If None, all samples in the dataset will be returned.
        If True, the first 80% examples are returned (train-val split). Otherwise, the last 20% examples are returned (test split).
    :type train: bool
    :param train_proportion: Proportion of the dataset that will be used for training. The rest will be used for validation.
    :type train_proportion: float
    :param precomputed_partitions: If True, precomputed (randomly and stratified) files dataset_train.dat and dataset_test.dat will be used.
        Otherwise, partitions will be computed from the same file as described by param "train".
    :type precomputed_partitions: bool
    :param num_workers: Number of workers used to load the data.
    :type num_workers: int
    :param pin_memory: If True, the data will be stored in the device/CUDA memory before returning them.
    :type pin_memory: bool
    '''
    # 1.-Prepare the datasets:

    X, y, header, class_names, boolean_categorical_vector = read_data(file_path)

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1 - train_proportion, random_state=random_state, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=random_state, stratify=y_train)

    return X_train, X_val, X_test, y_train, y_val, y_test, header, boolean_categorical_vector




def load_keel_dataset_pytorch(file_path, batch_size, train_proportion=0.8, num_workers=1, pin_memory=True, device=None, random_seed=33):
    '''
    :param file_path: Path to the dataset file.
    :type file_path: str
    :param batch_size: Number of samples in each minibatch.
    :type batch_size: int
    :param train: If None, all samples in the dataset will be returned.
        If True, the first 80% examples are returned (train-val split). Otherwise, the last 20% examples are returned (test split).
    :type train: bool
    :param train_proportion: Proportion of the dataset that will be used for training. The rest will be used for validation.
    :type train_proportion: float
    :param precomputed_partitions: If True, precomputed (randomly and stratified) files dataset_train.dat and dataset_test.dat will be used.
        Otherwise, partitions will be computed from the same file as described by param "train".
    :type precomputed_partitions: bool
    :param num_workers: Number of workers used to load the data.
    :type num_workers: int
    :param pin_memory: If True, the data will be stored in the device/CUDA memory before returning them.
    :type pin_memory: bool
    '''
    X_train, X_val, X_test, y_train, y_val, y_test, header, boolean_categorical_vector = load_KeelDatasetPytorch(file_path, batch_size, train_proportion, random_state=random_seed)
    columns = list(X_train.columns)
    # Normalize scaling the data:
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    # Scale only the non-categorical columns:
    boolean_categorical_vector = np.array(boolean_categorical_vector)
    X_train = X_train.values
    X_val = X_val.values
    X_test = X_test.values

    if sum(~boolean_categorical_vector) > 0:
        X_train[:, ~boolean_categorical_vector] = scaler.fit_transform(X_train[:, ~boolean_categorical_vector])
        X_val[:, ~boolean_categorical_vector] = scaler.transform(X_val[:, ~boolean_categorical_vector])
        X_test[:, ~boolean_categorical_vector] = scaler.transform(X_test[:, ~boolean_categorical_vector])

    trainDataset = transform_to_pytorch_dataset(X_train, y_train, device=device)
    valDataset = transform_to_pytorch_dataset(X_val, y_val, device=device)
    testDataset = transform_to_pytorch_dataset(X_test, y_test, device=device)

    trainDataLoader = DataLoader(trainDataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
    valDataLoader = DataLoader(valDataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
    testDataLoader = DataLoader(testDataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)

    return trainDataLoader, valDataLoader, testDataLoader, header, columns, boolean_categorical_vector


def load_keel(dataset_name):
    #file_path = os.path.join("../keel_datasets", dataset_name, dataset_name + ".dat")
    trainDataLoader, valDataLoader, testDataLoader, auxData, feat_names, boolean_categorical_vector = load_keel_dataset_pytorch(dataset_name, batch_size=32, train_proportion=0.8, num_workers=0, pin_memory=True)
    X_train = trainDataLoader.dataset.X.cpu().detach().numpy()
    X_train = pd.DataFrame(X_train, columns=feat_names)
    y_train = trainDataLoader.dataset.y.cpu().detach().numpy()
    X_val = valDataLoader.dataset.X.cpu().detach().numpy()
    X_val = pd.DataFrame(X_val, columns=feat_names)
    y_val = valDataLoader.dataset.y.cpu().detach().numpy()
    X_test = testDataLoader.dataset.X.cpu().detach().numpy()
    X_test = pd.DataFrame(X_test, columns=feat_names)
    y_test = testDataLoader.dataset.y.cpu().detach().numpy()

    X_train = X_train.fillna(X_train.mean())
    X_val = X_val.fillna(X_val.mean())
    X_test = X_test.fillna(X_test.mean())

    X_total = pd.concat([X_train, X_val, X_test], axis=0)
    X_total = X_total.fillna(X_total.mean())
    y_total = np.concatenate([y_train, y_val, y_test], axis=0)
    
    translator = X_total.columns
    different_labels = np.unique(y_total)
    label_dict = {different_labels[i]:i for i in range(len(different_labels))}
    return X_total, y_total, label_dict, translator


def predefined_dataset(name, binary_y=False):
    """
    Define how to read specific datasets and return structured X and Y data.
    
    Args
        name (str): the name of the dataset to read.
        binary_y (bool): if True, force the dataset to only have two classes.
        
    Returns
        table_X (DataFrame): instances, values can be strings or numbers.
        table_Y (DataFrame): labels, values can be strings or numbers.
        categorical_cols (list): A list of column names that are categorical data. 
        numerical_cols (list): A list of column names that are numerical data.
    """
    
    dir_path = os.path.dirname(os.path.realpath(__file__)) # .py
    
    ### UCI datasets
    if name == 'adultOG':
        # https://archive.ics.uci.edu/ml/datasets/adult
        # X dim: (30162, 14)
        # Y counts: {'<=50K': 22654, '>50K': 7508}
        table = pd.read_csv(dir_path + '/adult/adult.data', header=0, na_values='?', skipinitialspace=True).dropna()
        table_X = table.iloc[:, :-1].copy()
        table_Y = table.iloc[:, -1].copy()
        categorical_cols = None
        numerical_cols = None
        
    elif name == 'magic':
        # http://archive.ics.uci.edu/ml/datasets/MAGIC+GAMMA+Telescope
        # X dim: (19020, 10/90)
        # Y counts: {'g': 12332, 'h': 6688}
        table = pd.read_csv(dir_path + '/magic/magic04.data', header=0, na_values='?', skipinitialspace=True).dropna()
        table_X = table.iloc[:, :-1].copy()
        table_Y = table.iloc[:, -1].copy()
        categorical_cols = None
        numerical_cols = None
    
    ### OpenML datasets
    elif name == 'house':
        # https://www.openml.org/d/821
        # X dim: (22784, 16/132)
        # Y counts: {'N': 6744, 'P': 16040}
        table = pd.read_csv(dir_path + '/house/house_16H.csv', header=0, skipinitialspace=True)
        table_X = table.iloc[:, :-1].copy()
        table_Y = table.iloc[:, -1].copy()
        categorical_cols = None
        numerical_cols = None
    
    ### Others
    elif name == 'heloc':
        # https://community.fico.com/s/explainable-machine-learning-challenge?tabset-3158a=2&tabset-158d9=3
        # X dim: (2502, 23)
        # Y counts: {'Bad': 1560, 'Good': 942}
        table = pd.read_csv(dir_path + '/heloc/heloc_dataset_v1.csv', header=0, na_values=['-7', '-8', '-9'], skipinitialspace=True)#.dropna()
        table_X = table.iloc[:, 1:].copy()
        table_Y = table.iloc[:, 0].copy()
        categorical_cols = None
        numerical_cols = None
        
    else:
        # It is a keel dataset
        dataset_path = os.path.join('../keel_datasets-master/', name, name + '.dat')
        table_X, table_Y, _, _ = load_keel(dataset_path)
        categorical_cols = None
        numerical_cols = None

    return table_X, table_Y, categorical_cols, numerical_cols

def transform_dataset(name, method='ordinal', negations=False, labels='ordinal'):
    """
    Transform values in datasets (from predefined_dataset) into real numbers or binary numbers.
    
    Args
        name (str): the name of the dataset.
        method (str): specify how the instances are encoded:
            'origin': encode categorical features as integers and leave the numerical features as they are (float).
            'ordinal': encode all features as integers; numerical features are discretized into intervals.
            'onehot': one-hot encode the integer features transformed using 'ordinal' method.
            'onehot-compare': one-hot encode the categorical features just like how they are done in 'onehot' method; 
                one-hot encode numerical features by comparing them with different threhsolds and encode 1 if they are smaller than threholds. 
        negations (bool): whether append negated binary features; only valid when method is 'onehot' or 'onehot-compare'. 
        labels (str): specify how the labels are transformed.
            'ordinal': output Y is a 1d array of integer values ([0, 1, 2, ...]); each label is an integer value.
            'binary': output Y is a 1d array of binary values ([0, 1, 0, ...]); each label is forced to be a binary value (see predefined_dataset).
            'onehot': output Y is a 2d array of one-hot encoded values ([[0, 1, 0], [1, 0, 0], [0, 0, 1]]); each label is a one-hot encoded 1d array.
    
    Return
        X (DataFrame): 2d float array; transformed instances.
        Y (np.array): 1d or 2d (labels='onehot') integer array; transformed labels;.
        X_headers (list|dict): if method='ordinal', a dict where keys are features and values and their categories; otherwise, a list of binarized features.
        Y_headers (list): the names of the labels, indexed by the values in Y.
    """
    
    METHOD = ['origin', 'ordinal', 'onehot', 'onehot-compare']
    LABELS = ['ordinal', 'binary', 'onehot']
    if method not in METHOD:
        raise ValueError(f'method={method} is not a valid option. The options are {METHOD}')
    if labels not in LABELS:
        raise ValueError(f'labels={labels} is not a valid option. The options are {LABELS}')
    
    table_X, table_Y, categorical_cols, numerical_cols = predefined_dataset(name, binary_y=labels == 'binary')

    # By default, columns with object type are treated as categorical features and rest are numerical features
    # All numerical features that have fewer than 5 unique values should be considered as categorical features
    if categorical_cols is None:
        categorical_cols = list(table_X.columns[(table_X.dtypes == np.dtype('O')).to_numpy().nonzero()[0]])
    if numerical_cols is None:
        numerical_cols = [col for col in table_X.columns if col not in categorical_cols and np.unique(table_X[col].to_numpy()).shape[0] > 5]
        categorical_cols = [col for col in table_X.columns if col not in numerical_cols]
            
    # Fill categorical nan values with most frequent value and numerical nan values with the mean value
    if len(categorical_cols) != 0:
        imp_cat = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
        table_X[categorical_cols] = imp_cat.fit_transform(table_X[categorical_cols])
    if len(numerical_cols) != 0:
        imp_num = SimpleImputer(missing_values=np.nan, strategy='mean')
        table_X[numerical_cols] = imp_num.fit_transform(table_X[numerical_cols])
        
    if np.nan in table_X or np.nan in table_Y:
        raise ValueError('Dataset should not have nan value!')
        
    # Encode instances
    X = table_X.copy()
    
    col_categories = []
    if method in ['origin', 'ordinal'] and len(categorical_cols) != 0:
        # Convert categorical strings to integers that represent different categories
        ord_enc = OrdinalEncoder()
        X[categorical_cols] = ord_enc.fit_transform(X[categorical_cols])
        col_categories = {col: list(categories) for col, categories in zip(categorical_cols, ord_enc.categories_)}

    col_intervals = []
    if method in ['ordinal', 'onehot'] and len(numerical_cols) != 0:
        # Discretize numerical values to integers that represent different intervals
        kbin_dis = KBinsDiscretizer(encode='ordinal', strategy='kmeans')
        X[numerical_cols] = kbin_dis.fit_transform(X[numerical_cols])
        col_intervals = {col: [f'({intervals[i]:.2f}, {intervals[i+1]:.2f})' for i in range(len(intervals) - 1)] for col, intervals in zip(numerical_cols, kbin_dis.bin_edges_)}

        if method in ['onehot']:
            # Make numerical values to interval strings so that FeatureBinarizer can process them as categorical values
            for col in numerical_cols:
                X[col]  = np.array(col_intervals[col]).astype('object')[X[col].astype(int)]

    if method in ['onehot', 'onehot-compare']:
        # One-hot encode categorical values and encode numerical values by comparing with thresholds
        fb = FeatureBinarizer(colCateg=categorical_cols, negations=negations)
        X = fb.fit_transform(X)
    
    if method in ['origin']:
        # X_headers is a list of features
        X_headers = [column for column in X.columns]
    if method in ['ordinal']:
        # X_headers is a dict where keys are features and values and their categories
        X_headers = {col: col_categories[col] if col in col_categories else col_intervals[col] for col in table_X.columns}
    else:
        # X_headers is a list of binarized features
        X_headers = ["".join(map(str, column)) for column in X.columns]
        
    if method not in ['origin']:
        X = X.astype(int)
    
    # Encode labels
    le = LabelEncoder()
    Y = le.fit_transform(table_Y).astype(int)
    Y_headers = le.classes_
    if labels == 'onehot':
        lb = LabelBinarizer()
        Y = lb.fit_transform(Y)
    
    return X, Y, X_headers, Y_headers

def split_dataset(X, Y, test=0.2, shuffle=None):    
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test, random_state=shuffle)
    
    return X_train, X_test, Y_train, Y_test

def kfold_dataset(X, Y, k=5, shuffle=None):
    kf = StratifiedKFold(n_splits=k, shuffle=bool(shuffle), random_state=shuffle)
    datasets = [(X.iloc[train_index], X.iloc[test_index], Y[train_index], Y[test_index]) 
                for train_index, test_index in kf.split(X, Y if len(Y.shape) == 1 else Y.argmax(1))]
    
    return datasets

def nested_kfold_dataset(X, Y, outer_k=5, inner_k=5, shuffle=None):
    inner_kf = StratifiedKFold(n_splits=inner_k, shuffle=bool(shuffle), random_state=shuffle)
    
    datasets = []
    for dataset in kfold_dataset(X, Y, k=outer_k, shuffle=shuffle):
        X_train_valid, X_test, Y_train_valid, Y_test = dataset
        
        nested_datasets = []
        for train_index, valid_index in inner_kf.split(
            X_train_valid, Y_train_valid if len(Y.shape) == 1 else Y_train_valid.argmax(1)):
            X_train = X.iloc[train_index]
            X_valid = X.iloc[valid_index]
            Y_train = Y[train_index]
            Y_valid = Y[valid_index]
            nested_datasets.append([X_train, X_valid, Y_train, Y_valid])
        datasets.append([X_train_valid, X_test, Y_train_valid, Y_test, nested_datasets])
    
    return datasets