import pmdarima.arima as pm
import numpy as np




def AutoARIMA(inputArray, forecast_steps):
    
    model = pm.AutoARIMA(start_p=1, d=None, start_q=1, max_p=99, max_d=16, max_q=99,
                         trace=True, suppress_warnings=True, stepwise=False,
                         seasonal=True, maxiter=100)
    predValue = model.fit_predict(y= inputArray, n_periods= forecast_steps)
    
    return predValue




if __name__ == "__main__" :

    testArray = np.array([i for i in range(16)])
    print(AutoARIMA(testArray, 4))