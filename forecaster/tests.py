from django.test import TestCase
import os
from os.path import exists
from django.conf import settings
from .LSTM_Serealizer import LSTM_Serealizer
from keras.models import load_model
from unittest import skip
from .forecastingHoursSerealizer import HoursSerealizer
from datetime import date
import time

# Create your tests here.


def forecasterHelper(lags, measurementsDict, filePath):
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


class ForecastingTests(TestCase):

    @skip("Its tested")
    def test_D001_sensorLSTModelExistence(self):
        # model = load_model(settings.BASE_DIR,+'./static/lstmModels/D001/1H_Forecast/1H_Forecast')
        model = load_model(os.path.join(
            settings.BASE_DIR, 'static\lstmModels\D003\\1H_Forecast\\1H_ForecastModel_9_SizeWindow'))

        self.assertIsNotNone(model)

    @skip("Its tested")
    def test_D001_sigleSensorLSTModelForecast(self):
        model = load_model(os.path.join(
            settings.BASE_DIR, 'static\lstmModels\D001\\5H_Forecast\\5H_ForecastModel_7_SizeWindow'))
        prediction = model.predict([[
            [21.24499, 375.18, 86,  99.924, 110],
            [20.9063633, 399.8299999, 72.7272727, 99.913181, 113],
            [20.851999, 467.5660000000001, 71.3, 99.91199, 115],
            [20.719090, 438.38272, 71.63636, 99.90709, 116.45454],
            [20.70444444444444, 424.17555, 70.555555, 99.931444, 115.6666666],
            [20.662499, 336.94, 67.83333333333333, 99.97834, 96],
            [20.8654, 199.89727, 68.818181, 100.287727, 92.9090],
            [20.8654, 199.89727, 68.818181, 100.287727, 92.9090],
            [20.8654, 199.89727, 68.818181, 100.287727, 92.9090]
        ]])

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
                        forecast = forecasterHelper(9, measurementsDict, 'hour1', os.path.join(
                            settings.BASE_DIR, 'static\lstmModels\D001\\1H_Forecast\\1H_ForecastModel_9_SizeWindow'))
                        forecastDict['hour1'] = forecast

                case "hour2":
                    if shouldForecast == True:
                        forecast = forecasterHelper(7, measurementsDict, 'hour2', os.path.join(
                            settings.BASE_DIR, 'static\lstmModels\D001\\2H_Forecast\\2H_ForecastModel_7_SizeWindow'))
                        forecastDict['hour2'] = forecast

                case "hour3":
                    if shouldForecast == True:
                        forecast = forecasterHelper(8, measurementsDict, 'hour3', os.path.join(
                            settings.BASE_DIR, 'static\lstmModels\D001\\3H_Forecast\\3H_ForecastModel_8_SizeWindow'))
                        forecastDict['hour3'] = forecast

                case "hour4":
                    if shouldForecast == True:
                        forecast = forecasterHelper(9, measurementsDict, 'hour4', os.path.join(
                            settings.BASE_DIR, 'static\lstmModels\D001\\4H_Forecast\\4H_ForecastModel_9_SizeWindow'))
                        forecastDict['hour4'] = forecast

                case "hour5":
                    if shouldForecast == True:
                        forecast = forecasterHelper(7, measurementsDict, 'hour5', os.path.join(
                            settings.BASE_DIR, 'static\lstmModels\D001\\5H_Forecast\\5H_ForecastModel_7_SizeWindow'))
                        forecastDict['hour5'] = forecast

                case "hour6":
                    if shouldForecast == True:
                        forecast = forecasterHelper(9, measurementsDict, 'hour6', os.path.join(
                            settings.BASE_DIR, 'static\lstmModels\D001\\6H_Forecast\\6H_ForecastModel_9_SizeWindow'))
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
        measurementsSerealizer = LSTM_Serealizer(data=measurements,many=True)
        if measurementsSerealizer.is_valid():
            print(len(measurementsSerealizer.data))
            for i in range(len(measurementsSerealizer.data)):
                measurmentsDict.append(dict(measurementsSerealizer.data[i]))
            
            
            print(measurmentsDict)
            self.assertIsNotNone(measurmentsDict)


    def test_sensorDataFetch(self):
        nineHoursInMilliseconds = 32400000
        obj = time.gmtime(0)
        epoch = time.asctime(obj)
        print("The epoch is:",epoch)
        curr_time = round(time.time()*1000)
        print("Milliseconds since epoch:",curr_time)
        print("Milliseconds - 9 hours epoch:",
              curr_time - nineHoursInMilliseconds)
