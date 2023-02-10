from django.test import TestCase
import os
from os.path import exists
from django.conf import settings
from .MeasurementsSerializer import MeasurementsSerealizer
from keras.models import load_model
from unittest import skip
from .forecastingHoursSerealizer import HoursSerealizer
from datetime import date
import time
from .algorithms.lstm.lstmHandler import lstmSensorsModelsDetails
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from pandas.testing import assert_frame_equal
import numpy as np

# Create your tests here.

"""Helper functions section for testing"""


def forecaster(lags, measurementsDict, filePath):
    measurementsLagArray = [[]]  # necessary shape to make the forecasts
    measureArray = []

    for i in range(lags):  # range value is the model lag
        for measure in measurementsDict[i].values():
            measureArray.append(measure)

        measurementsLagArray[0].append(measureArray.copy())
        measureArray.clear()

    model = load_model(os.path.join(
        settings.BASE_DIR, filePath))
    forecast = model.predict(measurementsLagArray)
    measurementsLagArray.clear()

    return forecast[0][0]


def datasetScaler(df):
    scaler = StandardScaler()
    scaler.fit(df)

    normalized_df = pd.DataFrame(scaler.transform(
        df), columns=df.columns, index=df.index)

    return normalized_df, scaler


def inverse_scale_1d(array, scaler, column, data):
    col_scaler = StandardScaler().fit(data[[column]])
    inverse_scaled_value = col_scaler.inverse_transform(array.reshape(-1, 1))
    return inverse_scaled_value


def normalize_forecast_input_data(data, means, stds):
    """
    Normalize a single input to prepare it for prediction.

    Parameters:
    - data: a list of lists, where each list inside represents a row of data
    - means: a list of means for each column
    - stds: a list of standard deviations for each column

    Returns:
    - normalized_data: a list of lists, where each list inside represents a row of normalized data
    """
    normalized_data = []
    for i in range(len(data)):
        normalized_row = [(data[i][j] - means[j]) / stds[j]
                          for j in range(len(data[i]))]
        normalized_data.append(normalized_row)
    return [normalized_data]


def inverse_normalize_value(normalized_value, mean, std):
    original_value = normalized_value * std + mean
    return original_value


def inverse_forecast_normalize_data(normalized_data, means, stds):
    """
    Inverse normalize a single input to realize the prediction.

    Parameters:
    - data: a list of lists, where each list inside represents a row of data
    - means: a list of means for each column
    - stds: a list of standard deviations for each column

    Returns:
    - inverse_normalized_data: a list of lists, where each list inside represents a row of inverse normalized data
    """
    original_data = []
    for i in range(len(normalized_data)):
        if len(normalized_data[i]) != len(means) or len(normalized_data[i]) != len(stds):
            raise ValueError(
                "Length of normalized data row and means or stds must be equal.")
        original_row = [normalized_data[i][j] * stds[j] + means[j]
                        for j in range(len(normalized_data[i]))]
        original_data.append(original_row)
    return original_data


def df_to_X_y(df, hoursToPredict, windowSize):
    hoursToPredict = hoursToPredict - 1
    df_as_np = df.to_numpy()  # converts the dataframe to a numpy array
    # Initialized  arrays to append X and Y values
    X = []
    y = []
    for i in range(len(df_as_np)):
        if ((i + hoursToPredict + windowSize) < len(df_as_np)):
            # Takes values from i to i + win size
            row = [r for r in df_as_np[i:i+windowSize]]
            X.append(row)
            label = df_as_np[i + hoursToPredict + windowSize][1]
            y.append(label)
    return np.array(X), np.array(y)


def mean_absolute_error(df1, df2):
    """Calculate the mean() absolute error between two Pandas DataFrames.

    Arguments:
    df1 -- a Pandas DataFrame
    df2 -- a Pandas DataFrame

    Returns:
    mae -- a float representing the mean() absolute error
    """
    mae = (np.abs(df1 - df2)).mean()().mean()()
    return mae


def datasetCleaner(df):
    df['time'] = pd.to_datetime(df['time'], format='%Y-%m-%d %H:%M:%S')
    df.index = df['time']
    df = df.drop(['time'], axis=1)

    return df


"""End of helper functions"""


class ForecastingTests(TestCase):
    @skip("Its tested")
    def test_D001_sensorLSTModelExistence(self):
        # model = load_model(settings.BASE_DIR,+'./static/lstmModels/D001/1H_Forecast/1H_Forecast')
        model = load_model(os.path.join(
            settings.BASE_DIR, 'static/lstmModels/D003//1H_Forecast//1H_ForecastModel_9_SizeWindow'))

        self.assertIsNotNone(model)

    @skip("Its tested")
    def test_D001_sigleSensorLSTModelForecast(self):
        model = load_model(os.path.join(
            settings.BASE_DIR, 'static/lstmModels/D001//5H_Forecast//5H_ForecastModel_7_SizeWindow'))
        prediction = model.predict(
            [[
                [21.24499, 375.18, 86,  99.924, 110],
                [20.9063633, 399.8299999, 72.7272727, 99.913181, 113],
                [20.851999, 467.5660000000001, 71.3, 99.91199, 115],
                [20.719090, 438.38272, 71.63636, 99.90709, 116.45454],
                [20.70444444444444, 424.17555, 70.555555, 99.931444, 115.6666666],
                [20.662499, 336.94, 67.83333333333333, 99.97834, 96],
                [20.8654, 199.89727, 68.818181, 100.287727, 92.9090],
            ]]
        )

        print("forecasted radon level: ", prediction)
        self.assertIsNotNone(prediction)

    @skip("Its tested")
    def test_multipleSensorsLSTModelsForecast(self):
        hoursDict = {
            "hour1": True,
            "hour2": True,
            "hour3": True,
            "hour4": True,
            "hour5": True,
            "hour6": True
        }
        forecastDict = {}

        measurementsDict = [
            {
                "temperature": 22.2,
                "radon": 1021.21,
                "humidity": 46.6,
                "pressure": 100.02,
                "co2": 200.2
            },
            {
                "temperature": 22.2,
                "radon": 1021.21,
                "humidity": 46.6,
                "pressure": 100.02,
                "co2": 200.2
            },
            {
                "temperature": 22.2,
                "radon": 1021.21,
                "humidity": 46.6,
                "pressure": 100.02,
                "co2": 200.2
            },
            {
                "temperature": 22.2,
                "radon": 1021.21,
                "humidity": 46.6,
                "pressure": 100.02,
                "co2": 200.2
            },
            {
                "temperature": 22.2,
                "radon": 1021.21,
                "humidity": 46.6,
                "pressure": 100.02,
                "co2": 200.2
            },
            {
                "temperature": 22.2,
                "radon": 1021.21,
                "humidity": 46.6,
                "pressure": 100.02,
                "co2": 200.2
            },
            {
                "temperature": 22.2,
                "radon": 1021.21,
                "humidity": 46.6,
                "pressure": 100.02,
                "co2": 200.2
            },
            {
                "temperature": 22.2,
                "radon": 1021.21,
                "humidity": 46.6,
                "pressure": 100.02,
                "co2": 200.2
            },
            {
                "temperature": 22.2,
                "radon": 1021.21,
                "humidity": 46.6,
                "pressure": 100.02,
                "co2": 200.2
            },
        ]

        for key, shouldForecast in hoursDict.items():
            match key:
                case "hour1":
                    if shouldForecast == True:
                        forecast = forecaster(9, measurementsDict, os.path.join(
                            settings.BASE_DIR, 'static/lstmModels/D001//1H_Forecast//1H_ForecastModel_9_SizeWindow'))
                        forecastDict['hour1'] = forecast

                case "hour2":
                    if shouldForecast == True:
                        forecast = forecaster(7, measurementsDict, os.path.join(
                            settings.BASE_DIR, 'static/lstmModels/D001//2H_Forecast//2H_ForecastModel_7_SizeWindow'))
                        forecastDict['hour2'] = forecast

                case "hour3":
                    if shouldForecast == True:
                        forecast = forecaster(8, measurementsDict, os.path.join(
                            settings.BASE_DIR, 'static/lstmModels/D001//3H_Forecast//3H_ForecastModel_8_SizeWindow'))
                        forecastDict['hour3'] = forecast

                case "hour4":
                    if shouldForecast == True:
                        forecast = forecaster(9, measurementsDict, os.path.join(
                            settings.BASE_DIR, 'static/lstmModels/D001//4H_Forecast//4H_ForecastModel_9_SizeWindow'))
                        forecastDict['hour4'] = forecast

                case "hour5":
                    if shouldForecast == True:
                        forecast = forecaster(7, measurementsDict, os.path.join(
                            settings.BASE_DIR, 'static/lstmModels/D001//5H_Forecast//5H_ForecastModel_7_SizeWindow'))
                        forecastDict['hour5'] = forecast

                case "hour6":
                    if shouldForecast == True:
                        forecast = forecaster(9, measurementsDict, os.path.join(
                            settings.BASE_DIR, 'static/lstmModels/D001//6H_Forecast//6H_ForecastModel_9_SizeWindow'))
                        forecastDict['hour6'] = forecast

        print(forecastDict)
        self.assertIsNotNone(forecastDict)

    @skip("Its tested")
    def test_dataSerealizing(self):
        hours = {
            "hour1": True,
            "hour2": False,
            "hour3": True,
            "hour4": False,
            "hour5": True,
            "hour6": False
        }
        hoursSerealizer = HoursSerealizer(data=hours)
        print(hoursSerealizer)
        self.assertIsNotNone(hoursSerealizer)

    @skip("Its tested")
    def test_serealizedDataIteration(self):
        hours = {
            "hour1": True,
            "hour2": False,
            "hour3": True,
            "hour4": False,
            "hour5": True,
            "hour6": False
        }
        hoursSerealizer = HoursSerealizer(data=hours)
        if hoursSerealizer.is_valid():
            parsedHours = dict(hoursSerealizer.data)
            print(type(parsedHours))
            print(parsedHours)
            print(type(parsedHours))
            for value in hoursSerealizer.data.values():
                print(value)

    @skip("Its tested")
    def test_serealizedArrayDataIteration(self):
        measurements = [
            {
                "temperature": 22.2,
                "radon": 1021.21,
                "humidity": 46.6,
                "pressure": 100.02,
                "co2": 200.2
            },
            {
                "temperature": 22.2,
                "radon": 1021.21,
                "humidity": 46.6,
                "pressure": 100.02,
                "co2": 200.2
            },
            {
                "temperature": 22.2,
                "radon": 1021.21,
                "humidity": 46.6,
                "pressure": 100.02,
                "co2": 200.2
            },
            {
                "temperature": 22.2,
                "radon": 1021.21,
                "humidity": 46.6,
                "pressure": 100.02,
                "co2": 200.2
            },
            {
                "temperature": 22.2,
                "radon": 1021.21,
                "humidity": 46.6,
                "pressure": 100.02,
                "co2": 200.2
            },
            {
                "temperature": 22.2,
                "radon": 1021.21,
                "humidity": 46.6,
                "pressure": 100.02,
                "co2": 200.2
            },
            {
                "temperature": 22.2,
                "radon": 1021.21,
                "humidity": 46.6,
                "pressure": 100.02,
                "co2": 200.2
            },
            {
                "temperature": 22.2,
                "radon": 1021.21,
                "humidity": 46.6,
                "pressure": 100.02,
                "co2": 200.2
            },
            {
                "temperature": 22.2,
                "radon": 1021.21,
                "humidity": 46.6,
                "pressure": 100.02,
                "co2": 200.2
            }
        ]
        measurmentsDict = []
        measurementsSerealizer = MeasurementsSerealizer(
            data=measurements, many=True)
        if measurementsSerealizer.is_valid():
            print(len(measurementsSerealizer.data))
            for i in range(len(measurementsSerealizer.data)):
                measurmentsDict.append(dict(measurementsSerealizer.data[i]))

            print(measurmentsDict)
            self.assertIsNotNone(measurmentsDict)

    @skip("Its tested")
    def test_sensorDataFetch(self):

        nineHoursInMilliseconds = 32400000
        obj = time.gmtime(0)
        epoch = time.asctime(obj)
        print("The epoch is:", epoch)
        curr_time = round(time.time()*1000)
        print("Milliseconds since epoch:", curr_time)
        print("Milliseconds - 9 hours epoch:",
              curr_time - nineHoursInMilliseconds)

    @skip("Its tested")
    def test_getSensorsDetails(self):
        details = lstmSensorsModelsDetails()
        D001_details = details['D001']
        print(D001_details)
        print("Details: /n", D001_details)
        print("Details: /n", D001_details['D001'][0]['error'])

        self.assertIsNotNone(D001_details['D001'])

    @skip("Its tested")
    def test_modelOpeningWithConcat(self):
        D001_details = lstmSensorsModelsDetails()
        print("Details: /n", D001_details['D001'])
        print("Details: /n", D001_details['D001'][0]['error'])

        self.assertIsNotNone(D001_details['D001'])
        # model = load_model(settings.BASE_DIR,+'./static/lstmModels/D001/1H_Forecast/1H_Forecast')
        # model = load_model(os.path.join(
        #     settings.BASE_DIR, 'static/lstmModels/D003//1H_Forecast//1H_ForecastModel_', D001_details['D001'][0]['bestLag'] ,'_SizeWindow'))
        print(type(D001_details['D001'][0]['bestLag']))
        model = load_model(os.path.join(settings.BASE_DIR, "static/lstmModels/D003//1H_Forecast//1H_ForecastModel_{}_SizeWindow".format(
            D001_details['D001'][0]['bestLag'])))
        print(model)

        self.assertIsNotNone(model)


class NormalizationTests(TestCase):
    @skip("Its tested")
    def test_normalizationDatasetExistence(self):
        filePath = os.path.join(
            settings.BASE_DIR, 'static/datasets/interpolated_D003_data.csv')
        df = pd.read_csv(filePath, encoding='utf-8')
        self.assertIsNotNone(df)

    @skip("Its tested")
    def test_scaledDataVisualization(self):
        filePath = os.path.join(
            settings.BASE_DIR, 'static/datasets/interpolated_D003_data.csv')

        df = pd.read_csv(filePath, encoding='utf-8')
        df = datasetCleaner(df)

        normalized_df, scaler = datasetScaler(df)

        print(normalized_df)

    # @skip("Its tested")
    def test_reverseScaler(self):
        filePath = os.path.join(
            settings.BASE_DIR, 'static/datasets/interpolated_D003_data.csv')

        df = pd.read_csv(filePath, encoding='utf-8')
        df = datasetCleaner(df)

        normalized_df, scaler = datasetScaler(df)
        inverseDf = scaler.inverse_transform(normalized_df)
        print(normalized_df)
        print(pd.DataFrame(inverseDf))

        # Only index are different because of the scaling so its fine, they are equal!
        assert_frame_equal(df, pd.DataFrame(inverseDf))

    @skip("Its tested")
    def test_reverseScalePredictedData_vs_defaultPredictedDataOnCSV(self):
        filePath = os.path.join(
            settings.BASE_DIR, 'static/datasets/interpolated_D003_data.csv')
        nonNormalizedModel = load_model(os.path.join(
            settings.BASE_DIR, 'static/lstmModels/D001//1H_Forecast//1H_ForecastModel_6_SizeWindow'))
        normalizedModel = load_model(os.path.join(
            settings.BASE_DIR, 'static/lstmModels/D001//1H_Forecast//normalized//1H_ForecastModel_6_SizeWindow'))
        df = pd.read_csv(filePath, encoding='utf-8')
        df = datasetCleaner(df)

        X, y = df_to_X_y(df, 1, 6)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, shuffle=False)
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.25, shuffle=False)

        test_predictions = nonNormalizedModel.predict(X_test).flatten()

        print('normal predictions MAE')

        test_predictionsDf = pd.DataFrame(test_predictions)
        y_testDf = pd.DataFrame(y_test)
        print(mean_absolute_error(test_predictionsDf, y_testDf))
        # print(test_predictionsDf)
        pd.DataFrame(data={"Test predictions": test_predictionsDf[0], "Actual": y_testDf[0]}).to_csv(
            "normal.csv")

        normalized_df, scaler = datasetScaler(df)

        X, y = df_to_X_y(normalized_df, 1, 6)
        X_train_inv, X_test_inv, y_train_inv, y_test_inv = train_test_split(
            X, y, test_size=0.2, shuffle=False)
        X_train_inv, X_val_inv, y_train_inv, y_val_inv = train_test_split(
            X_train_inv, y_train_inv, test_size=0.25, shuffle=False)

        test_predictions_inv = normalizedModel.predict(X_test_inv).flatten()
        inversedScale = inverse_scale_1d(
            test_predictions_inv, scaler, 'Rn', df,)
        inverse_y_test = inverse_scale_1d(y_test_inv, scaler, 'Rn', df,)

        print('--------------------------------')
        print('inv predictions MAE')
        inverseDataDf = pd.DataFrame(inversedScale.flatten())
        inverse_y_trainDf = pd.DataFrame(inverse_y_test.flatten())
        print(mean_absolute_error(inverseDataDf, inverse_y_trainDf))
        pd.DataFrame(data={"Test predictions": inverseDataDf[0], "Actual": inverse_y_trainDf[0]}).to_csv(
            "reverse.csv")

    @skip("Its tested")
    def test_scaleSinglePredictionAndForecast(self):
        measurements = [[
            [21.24499, 375.18, 86,  99.924, 110],
            [20.9063633, 399.8299999, 72.7272727, 99.913181, 113],
            [20.851999, 467.5660000000001, 71.3, 99.91199, 115],
            [20.719090, 438.38272, 71.63636, 99.90709, 116.45454],
            [20.70444444444444, 424.17555, 70.555555, 99.931444, 115.6666666],
            [20.662499, 336.94, 67.83333333333333, 99.97834, 96],
        ]]

        filePath = os.path.join(
            settings.BASE_DIR, 'static/datasets/interpolated_D003_data.csv')
        df = pd.read_csv(filePath, encoding='utf-8')
        df = datasetCleaner(df)

        model = load_model(os.path.join(
            settings.BASE_DIR, 'static/lstmModels/D001//1H_Forecast//1H_ForecastModel_6_SizeWindow'))

        means = [df['T'].mean(), df['Rn'].mean(), df['H'].mean(),
                 df['P'].mean(), df['CO2'].mean()]
        stds = [df['T'].std(), df['Rn'].std(), df['H'].std(),
                df['P'].std(), df['CO2'].std()]

        normalizedInputForecastSample = normalize_forecast_input_data(
            measurements[0], means, stds)
        print(normalizedInputForecastSample)

        print('----------------------------------------------------------/n')

        inverseInputForecastSample = inverse_forecast_normalize_data(
            normalizedInputForecastSample, means, stds)
        print(inverseInputForecastSample)

    @skip("Its tested")
    def test_compareNormalizedAndNonNormalizedForecastOutput(self):
        measurements = [[
            [21.24499, 375.18, 86,  99.924, 110],
            [20.9063633, 399.8299999, 72.7272727, 99.913181, 113],
            [20.851999, 467.5660000000001, 71.3, 99.91199, 115],
            [20.719090, 438.38272, 71.63636, 99.90709, 116.45454],
            [20.70444444444444, 424.17555, 70.555555, 99.931444, 115.6666666],
            [20.662499, 336.94, 67.83333333333333, 99.97834, 96],
        ]]
        
        [[
          [22.2, 1021.21, 46.6, 100.02, 200.2], 
          [22.2, 1021.21, 46.6, 100.02, 200.2],
          [22.2, 1021.21, 46.6, 100.02, 200.2],
          [22.2, 1021.21, 46.6, 100.02, 200.2], 
          [22.2, 1021.21, 46.6, 100.02, 200.2], 
          [22.2, 1021.21, 46.6, 100.02, 200.2]
          ]]
        means = [df['T'].mean(), df['Rn'].mean(), df['H'].mean(),
                 df['P'].mean(), df['CO2'].mean()]
        stds = [df['T'].std(), df['Rn'].std(), df['H'].std(),
                df['P'].std(), df['CO2'].std()]


        filePath = os.path.join(settings.BASE_DIR, 'static/datasets/interpolated_D003_data.csv')
        df = pd.read_csv(filePath, encoding='utf-8')
        df = datasetCleaner(df)

        nonNormalizedModel = load_model(os.path.join(
            settings.BASE_DIR, 'static/lstmModels/D001//1H_Forecast//1H_ForecastModel_6_SizeWindow'))

        normalizedModel = load_model(os.path.join(
            settings.BASE_DIR, 'static/lstmModels/D001//1H_Forecast//normalized//1H_ForecastModel_6_SizeWindow'))

        nonNormalizedForecast = nonNormalizedModel.predict(measurements)
        print(nonNormalizedForecast)

        normalizedForecast = normalizedModel.predict(
            normalize_forecast_input_data(measurements[0], means, stds))
        print("Normalized ", normalizedForecast)
        print("Denormalized ", inverse_normalize_value(
            normalizedForecast, means[1], stds[1]))

    @skip("Its tested")
    def test_prints(self):
        
        filePath = os.path.join(settings.BASE_DIR, 'static/datasets/interpolated_D001_data.csv')
        df = pd.read_csv(filePath, encoding='utf-8')
        df = datasetCleaner(df)

        X, y = df_to_X_y(df, 1, 6)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,shuffle=False)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25,shuffle=False) 


        print(pd.DataFrame(y_test).head(30))
