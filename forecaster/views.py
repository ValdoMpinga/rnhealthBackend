from django.shortcuts import render
from .MeasurementsSerealizer import MeasurementsSerealizer
from .forecastingHoursSerealizer import HoursSerealizer
from .targetSensorSerealizer import TargetSensorSerealizer
from django.views import View
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
import os
from django.conf import settings
from keras.models import load_model
from .algorithms.lstm.lstmHandler import lstmSensorsModelsDetails
from .algorithms.biLstm.biLstmHandler import biLstmSensorsModelsDetails
import tensorflow as tf


class ForecastViewClass(View):
    targetSensor = None
    FORECASTING_HOURS = 6

    @api_view(['POST'])
    def targetSensorView(request):
        serealizedTargetSensor = TargetSensorSerealizer(data=request.data)

        if serealizedTargetSensor.is_valid():
            ForecastViewClass.targetSensor = serealizedTargetSensor
            return Response(status=status.HTTP_200_OK)
        else:
            return Response(status=status.HTTP_500_INTERNAL_SERVER_ERROR)

    @api_view(['POST'])
    # @tf.function(reduce_retracing=True)
    def lstmForecastView(request):

        forecasts = []   
        sensorsDetails = lstmSensorsModelsDetails()
        targetSensorDetails = sensorsDetails[ForecastViewClass.targetSensor.data['targetSensor']]

        serealizedMeasurements = MeasurementsSerealizer(
            data=request.data, many=True)

        if serealizedMeasurements.is_valid():

            print("LSTM data is valid", os.getcwd())

            measurements = serealizerListToDict(serealizedMeasurements)
            
            for hour in range(ForecastViewClass.FORECASTING_HOURS):
                filePath = os.path.join(
                settings.BASE_DIR, "static/lstmModels/{}//{}H_Forecast//{}H_ForecastModel_{}_SizeWindow".format(ForecastViewClass.targetSensor.data['targetSensor'], (hour+1),(hour+1),targetSensorDetails[hour]['bestLag']))

                forecast = forecaster(targetSensorDetails[hour]['bestLag'], measurements, filePath)
                forecasts.append(
                                {
                                    'hour': 'Hour {}'.format(hour+1),
                                    'LSTM_Forecast': forecast,
                                    'error': targetSensorDetails[hour]['error']
                                }

                            )

            return Response(forecasts, status=status.HTTP_200_OK)
        else:
            print("LSTM data is invalid", os.getcwd())
            return Response(status=status.HTTP_400_BAD_REQUEST)

    @api_view(['POST'])
    # @tf.function(reduce_retracing=True)
    def bi_lstmForecastView(request):

        forecasts = []
        sensorsDetails = biLstmSensorsModelsDetails()
        targetSensorDetails = sensorsDetails[ForecastViewClass.targetSensor.data['targetSensor']]

        serealizedMeasurements = MeasurementsSerealizer(
            data=request.data, many=True)

        if serealizedMeasurements.is_valid():

            print("BI LSTM data is valid", os.getcwd())

            measurements = serealizerListToDict(serealizedMeasurements)
            
            for hour in range(ForecastViewClass.FORECASTING_HOURS):
                filePath = os.path.join(
                settings.BASE_DIR, "static/bi_lstmModels/{}//{}H_Forecast//{}H_ForecastModel_{}_SizeWindow".format(ForecastViewClass.targetSensor.data['targetSensor'], (hour+1),(hour+1),targetSensorDetails[hour]['bestLag']))

                forecast = forecaster(targetSensorDetails[hour]['bestLag'], measurements, filePath)
                forecasts.append(
                                {
                                    'hour': 'Hour {}'.format(hour+1),
                                    'biLSTM_Forecast': forecast,
                                    'error': targetSensorDetails[hour]['error']
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
