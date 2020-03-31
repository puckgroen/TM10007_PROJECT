# Download the necessary packages
import numpy as np
# Classifiers
from sklearn import model_selection

# Data loading function
from hn.load_data import load_data

data = load_data()
print(f'The number of samples: {len(data.index)}')
print(f'The number of features: {len(data.columns)}')
print(data.keys())
Y = data['label']
print(Y)

# The data has 160 features and 113 samples/subjects. The labels are given as either T12 (low) or T34 (high).
# The aim of this study is to predict the T-stage (high/low) in patients with H&N cancer based on features, 
# extracted from CT. A good performance on this dataset would be above 70% mean accuracy.

# Split the dataset in design and test set. Later on the design set should make use of cross-validation? 
X_train, X_test, y_train, y_test = model_selection.train_test_split(data, Y, test_size=0.2, stratify=Y)
print(len(X_test))
print(len(X_train))

# Continue with the train dataset and use a 4 fold cross-validation 
cv_4fold = model_selection.StratifiedKFold(n_splits=4)

# Loop over the folds
for validation_index, test_index in cv_4fold.split(X_train, y_train):
    # For now I had to rearrange the dictionary of X_train to a list, otherwise we cannot index it. 
    # Not sure if this is the perfect way but just to make it visual, you can see the validation sizes printed. This seems to be ok!
    print('Validation size in current fold =')
    array = np.array(list(X_train.items()))
    
    # Split the data properly
    X_validation = array[validation_index]
    
    print(len(X_validation))
    y_validation = y_train[validation_index]
    
    X_test = array[test_index]
    y_test = y_train[test_index]