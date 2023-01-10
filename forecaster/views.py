from django.shortcuts import render
from .algorithms.lstm.lstmHandler import LSTM_forecaster
from .MeasurementsSerealizer import MeasurementsSerealizer
from .forecastingHoursSerealizer import HoursSerealizer
from django.views import View
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
import os
from django.conf import settings
from keras.models import load_model

class ForecastViewClass(View):
    hours = {}

    @api_view(['POST'])
    def forecastingHoursView(request):
        serealizedHours = HoursSerealizer(data=request.data)

        if serealizedHours.is_valid():
            ForecastViewClass.hours = serealizedHours
            print('submitted data: ', ForecastViewClass.hours.data)
            return Response(status=status.HTTP_200_OK)
        else:
            return Response(status=status.HTTP_500_INTERNAL_SERVER_ERROR)

    @api_view(['POST'])
    def forecastView(request):

        forecasts = []

        serealizedMeasurements = MeasurementsSerealizer(
            data=request.data, many=True)

        if serealizedMeasurements.is_valid():

            print("LSTM data is valid", os.getcwd())

            lstmDataDic = serealizerListToDict(serealizedMeasurements)
            parsedHours = dict(ForecastViewClass.hours.data)

            for key, value in parsedHours.items():
                match key:
                    case "hour1":
                        if value == True:
                            forecast = forecaster(9, lstmDataDic, os.path.join(
                                settings.BASE_DIR, 'static\lstmModels\D001\\1H_Forecast\\1H_ForecastModel_9_SizeWindow'))
                            forecasts.append(
                                {
                                    'hour': 'Hour 1',
                                    'LSTM_Forecast': forecast
                                }

                            )

                    case "hour2":
                        if value == True:
                            forecast = forecaster(7, lstmDataDic, os.path.join(
                                settings.BASE_DIR, 'static\lstmModels\D001\\2H_Forecast\\2H_ForecastModel_7_SizeWindow'))
                            forecasts.append(
                                {
                                    'hour': 'Hour 2',
                                    'LSTM_Forecast': forecast
                                }

                            )

                    case "hour3":
                        if value == True:
                            forecast = forecaster(8, lstmDataDic, os.path.join(
                                settings.BASE_DIR, 'static\lstmModels\D001\\3H_Forecast\\3H_ForecastModel_8_SizeWindow'))
                            forecasts.append(
                                {
                                    'hour': 'Hour 3',
                                    'LSTM_Forecast': forecast
                                }

                            )
                            
                    case "hour4":
                        if value == True:
                            forecast = forecaster(9, lstmDataDic,  os.path.join(
                                settings.BASE_DIR, 'static\lstmModels\D001\\4H_Forecast\\4H_ForecastModel_9_SizeWindow'))
                            forecasts.append(
                                {
                                    'hour': 'Hour 4',
                                    'LSTM_Forecast': forecast
                                }

                            )

                    case "hour5":
                        if value == True:
                            forecast = forecaster(7, lstmDataDic, os.path.join(
                                settings.BASE_DIR, 'static\lstmModels\D001\\5H_Forecast\\5H_ForecastModel_7_SizeWindow'))
                            forecasts.append(
                                {
                                    'hour': 'Hour 5',
                                    'LSTM_Forecast': forecast
                                }

                            )

                    case "hour6":
                        if value == True:
                            forecast = forecaster(9, lstmDataDic, os.path.join(
                                settings.BASE_DIR, 'static\lstmModels\D001\\6H_Forecast\\6H_ForecastModel_9_SizeWindow'))
                            forecasts.append(
                                {
                                    'hour': 'Hour 6',
                                    'LSTM_Forecast': forecast
                                }

                            )

            return Response(forecasts, status=status.HTTP_200_OK)
        else:
            print("LSTM data is invalid", os.getcwd())
            return Response(status=status.HTTP_400_BAD_REQUEST)


def forecaster(lags, serealizedMeasurements, AI_modelFilePath):
    measurementsLagArray = [[]]  # necessary shape to make the forecasts
    measureArray = []

    for i in range(lags):  # range value is the model lag
        for measure in serealizedMeasurements[i].values():
            measureArray.append(measure)

        measurementsLagArray[0].append(measureArray.copy())
        measureArray.clear()

    model = load_model(os.path.join(
        settings.BASE_DIR, AI_modelFilePath))
    forecast = model.predict(measurementsLagArray)
    measurementsLagArray.clear()

    return forecast[0][0]


def serealizerListToDict(list):
    measurmentsDict = []

    try:
        for i in range(len(list.data)):
            print(i, " -> ", dict(list.data[i]))
            measurmentsDict.append(dict(list.data[i]))

        return measurmentsDict

    except:
        print("Something went wrong on serealizerListToDict function ")
