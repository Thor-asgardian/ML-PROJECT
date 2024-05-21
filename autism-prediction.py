import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import gc

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

train.shape, test.shape

train.isnull().sum()

test.isnull().sum()

train.info()

train.describe()

del test['age_desc']
del train['age_desc']

obj_cols = []

for i in train.columns:
    if train[i].dtype == 'object':
        obj_cols.append(i)
        
obj_cols

from sklearn import preprocessing
le = preprocessing.LabelEncoder()

for col in obj_cols:
    train[col] = le.fit_transform(train[col])
    test[col] = le.transform(test[col])
    
train.info()

features = ['A1_Score', 'A2_Score', 'A3_Score', 'A4_Score', 'A5_Score', 'A6_Score', 'A7_Score', 'A8_Score', 'A9_Score', 
            'age','gender','ethnicity', 'jaundice', 'austim', 'contry_of_res', 'used_app_before', 'result', 'relation']

feat_eng = ['A1_Score', 'A2_Score', 'A3_Score', 'A4_Score', 'A5_Score', 'A6_Score', 'A7_Score', 'A8_Score', 'A9_Score', 'A10_Score', 'age', 'result']

def add_features(data):
    for col in feat_eng:
        feat = str(col) + '_mean'
        data[feat] = data[col].mean()
        feat = str(col) + '_sum'
        data[feat] = data[col].sum()
        feat = str(col) + '_std'
        data[feat] = data[col].std()
        
add_features(train)
add_features(test)

train.shape, test.shape

del train['ID']
X = train.copy()
y = X.pop('Class/ASD')

X.shape

from flaml import AutoML
automl = AutoML()

from sklearn.model_selection import train_test_split
_, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

automl.fit(X, y, task="regression", metric='rmse', time_budget=1800)

print('Best ML learner:', automl.best_estimator)
print('Best hyperparameter config:', automl.best_config)
print('Best RMSE on validation data: {0:.4g}'.format(automl.best_loss))
print('Training duration of best run: {0:.4g} s'.format(automl.best_config_train_time))

from sklearn.metrics import mean_squared_error, mean_absolute_error
pred = automl.predict(X_test)

mae = mean_absolute_error(y_test, pred)
mse = mean_squared_error(y_test, pred)

print("MAE: %.7f" % mae)
print("MSE: %.7f" % mse)

del test['ID']

gc.collect()

pred = automl.predict(test)

sub = pd.read_csv('sample_submission.csv')

sub['Class/ASD'] = pred
sub.to_csv('submission.csv', index=False)

print('Mean submission:', sub['Class/ASD'].mean(), '\n')
sub
