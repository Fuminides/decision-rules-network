
import torch
import numpy as np
import pandas as pd

from datasets.dataset import transform_dataset, kfold_dataset
from DRNet import train, DRNet


import sys
name = sys.argv[1]
X, Y, X_headers, Y_headers = transform_dataset(name, method='onehot-compare', negations=False, labels='binary')
datasets = kfold_dataset(X, Y, shuffle=1)
X_train, X_test, Y_train, Y_test = datasets[0]

train_set = torch.utils.data.TensorDataset(torch.Tensor(X_train.to_numpy()), torch.Tensor(Y_train))
test_set = torch.utils.data.TensorDataset(torch.Tensor(X_test.to_numpy()), torch.Tensor(Y_test))


# Train DR-Net
# Default learning rate (1e-2), and_lam (1e-2), and and_lam (1e-5) usually work the best. A large epochs number is necessary for a sparse rule set i.e 10000 epochs.
net = DRNet(train_set[:][0].size(1), 50, 1)
train(net, train_set, test_set=test_set, device='cuda', lr=1e-2, epochs=1000, batch_size=10,
      and_lam=1e-2, or_lam=1e-5, num_alter=500)

# Get accuracy and the rule net
accu = (net.predict(np.array(X_test)) == Y_test).mean()
rules = net.get_rules(X_headers)
print(f'Accuracy: {accu}, num rules: {len(rules)}, num conditions: {sum(map(len, rules))}')

average_rule_length = np.mean(([len(rule) for rule in rules]))
unique_conditions = set()
for rule in rules:
    for condition in rule:
        if isinstance(condition, tuple):
            condition = condition[0]
        unique_conditions.add(condition)

compelxity_score = np.log(len(rules) + average_rule_length + len(unique_conditions))
print(f'Complexity score: {compelxity_score}')
print(f'Number of unique conditions: {len(unique_conditions)}')    
print(f'Average rule length: {average_rule_length}')
print(f'Number of rules: {len(rules)}')

res = pd.DataFrame(
    {'Accuracy': [accu],
     'Complexity score': [compelxity_score],
     'Average rule length': [average_rule_length],
     'Number of unique conditions': [len(unique_conditions)],
     'Number of rules': [len(rules)]},
    index=[name])

res.to_csv('results/results_' + name + '.csv', sep=';')

