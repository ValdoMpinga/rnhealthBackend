from django.shortcuts import render
from .algorithms.lstm.lstmHandler import LSTM_forecaster
from .LSTM_Serealizer import LSTM_Serealizer
from .forecastingHoursSerealizer import HoursSerealizer
from django.http import HttpResponse
from django.views import View
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
import json
from konst import Constant,Constants
import os
from django.conf import settings
from keras.models import load_model

zahours = Constants(
    Constant(hour1="hour1"),
    Constant(hour2="hour2"),
    Constant(hour3="hour3"),
    Constant(hour4="hour4"),
    Constant(hour5="hour5"),
    Constant(hour6="hour6")
)

class ForecastView(View):
    hours = {}

    @api_view(['POST'])
    def setForecastingHours(request):
        hoursSerealizer = HoursSerealizer(data=request.data)

        if hoursSerealizer.is_valid():
            ForecastView.hours = hoursSerealizer

            return Response(status=status.HTTP_200_OK)
        else:
            return Response(status=status.HTTP_500_INTERNAL_SERVER_ERROR)

    @api_view(['POST'])
    def forecasts(request):

        forecastDict = {}
        lstm_Serealizer = LSTM_Serealizer(data=request.data, many=True)
        
        if lstm_Serealizer.is_valid():
            print("LSTM data is valid", os.getcwd())

            lstmDataDic = serealizerListToDict(lstm_Serealizer)
            parsedHours = dict(ForecastView.hours.data)
            
            for key, value in parsedHours.items():
                match key:
                    case "hour1":
                        if value == True:
                            forecast = forecasterHelper(9, lstmDataDic, os.path.join(
                                settings.BASE_DIR, 'static\lstmModels\D001\\1H_Forecast\\1H_ForecastModel_9_SizeWindow'))
                            forecastDict['Hour 1'] = forecast

                    case "hour2":
                        if value == True:
                            forecast = forecasterHelper(7, lstmDataDic, os.path.join(
                                settings.BASE_DIR, 'static\lstmModels\D001\\2H_Forecast\\2H_ForecastModel_7_SizeWindow'))
                            forecastDict['Hour 2'] = forecast

                    case "hour3":
                        if value == True:
                            forecast = forecasterHelper(8, lstmDataDic, os.path.join(
                                settings.BASE_DIR, 'static\lstmModels\D001\\3H_Forecast\\3H_ForecastModel_8_SizeWindow'))
                            forecastDict['Hour 3'] = forecast

                    case "hour4":
                        if value == True:
                            forecast = forecasterHelper(9, lstmDataDic,  os.path.join(
                                settings.BASE_DIR, 'static\lstmModels\D001\\4H_Forecast\\4H_ForecastModel_9_SizeWindow'))
                            forecastDict['Hour 4'] = forecast

                    case "hour5":
                        if value == True:
                            forecast = forecasterHelper(7, lstmDataDic, os.path.join(
                                settings.BASE_DIR, 'static\lstmModels\D001\\5H_Forecast\\5H_ForecastModel_7_SizeWindow'))
                            forecastDict['Hour 5'] = forecast

                    case "hour6":
                        if value == True:
                            forecast = forecasterHelper(9, lstmDataDic, os.path.join(
                                settings.BASE_DIR, 'static\lstmModels\D001\\6H_Forecast\\6H_ForecastModel_9_SizeWindow'))
                            forecastDict['Hour 6'] = forecast
            
            return Response(forecastDict, status=status.HTTP_200_OK)
        else:
            print("LSTM data is invalid", os.getcwd())
            return Response(status=status.HTTP_400_BAD_REQUEST)

        # forecast = ForecastSerealizer(LSTM_forecaster()).data
        # serealizedData = ForecastSerealizer(forecast)
        # return JsonResponse(serealizedData.data,safe=False)


def forecasterHelper(lags, lstm_Serealizer, filePath):
    measurementsLagArray = [[]]  # necessary shape to make the forecasts
    measureArray = []

    for i in range(lags):  # range value is the model lag
        for measure in lstm_Serealizer[i].values():
            measureArray.append(measure)

        measurementsLagArray[0].append(measureArray.copy())
        measureArray.clear()

    model = load_model(os.path.join(
        settings.BASE_DIR, filePath))
    forecast = model.predict(measurementsLagArray)
    measurementsLagArray.clear()

    return forecast[0][0]

def serealizerListToDict(list):
    measurmentsDict = []
    for i in range(len(list.data)):
        measurmentsDict.append(dict(list.data[i]))

    return measurmentsDict
