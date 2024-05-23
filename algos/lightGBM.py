import numpy as np
from lightgbm.sklearn import LGBMRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import GridSearchCV




def fit_lightGBM(inputs, labels, _search= False):
    cv_params = {
        'reg_lambda': np.arange(0, 5, 1),
        'max_depth': np.arange(4, 8, 1),
        'subsample': np.arange(0.8, 1, 0.1),
        'colsample_bytree': np.arange(0.8, 1, 0.1),
        'num_leaves': np.arange(26, 90, 8),
        }
    default_params = {
        'objective': 'regression',
        'n_estimators': 1200,
        'verbosity': -1,
        'boosting_type': 'gbdt',
        'reg_lambda': 0,
        'max_depth': 4,
        'subsample': 0.8, 
        'colsample_bytree': 0.8,
        'num_leaves': 26,
        'random_state': 42,
        'n_jobs': -1,
    }
    if _search is True :
        estimator = LGBMRegressor(**default_params)
        model = MultiOutputRegressor(GridSearchCV(estimator, cv_params, n_jobs= -1, scoring= 'neg_mean_absolute_error', verbose=1))
        model.fit(inputs, labels)
    else :
        model = MultiOutputRegressor(LGBMRegressor(**default_params))
        model.fit(inputs, labels)
    return model




def lightGBM_predict(model, x_test):
    return model.predict(x_test)