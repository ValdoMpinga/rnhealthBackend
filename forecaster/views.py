from django.shortcuts import render
from .MeasurementsSerializer import MeasurementsSerializer
from .forecastingHoursSerealizer import HoursSerealizer
from .targetSensorSerializer import TargetSensorSerializer
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
import pandas as pd

class ForecastViewClass(View):
    TARGET_SENSOR = None
    FORECASTING_HOURS = 6
    TARGET_SENSOR_MEANS =[]
    TARGET_SENSOR_STANDARD_DEV =[]


    @api_view(['POST'])
    def targetSensorView(request):
        # Initialize the serializer with the data from the request
        serializedTargetSensor = TargetSensorSerializer(data=request.data)

        # Check if the serialized data is valid
        if serializedTargetSensor.is_valid():
            # Get the validated target sensor data
            target_sensor = serializedTargetSensor.validated_data['targetSensor']
            
            # Set the TARGET_SENSOR attribute of the ForecastViewClass to the target sensor
            ForecastViewClass.TARGET_SENSOR = target_sensor
            
            # Construct the file path for the sensor data file
            sensor_data_file_path = os.path.join(
                settings.BASE_DIR, f"static/datasets/interpolated_{target_sensor}_data.csv")
            
            # Load the sensor dataset from the file
            sensor_dataset = pd.read_csv(sensor_data_file_path)
            
            # Call the setTargetForecastingSensorMeansAndStandardDeviation function with the sensor dataset
            setTargetForecastingSensorMeansAndStandardDeviation(sensor_dataset)

            # Return a 200 OK response if the serialized data is valid
            return Response(status=status.HTTP_200_OK)
        else:
            # Return a 400 Bad Request response with the errors from the serializer if the serialized data is not valid
            return Response(serializedTargetSensor.errors, status=status.HTTP_400_BAD_REQUEST)

    @api_view(['POST'])
    # @tf.function(reduce_retracing=True)
    def lstmForecastView(request):
        # Initialize an empty list to store the forecasts
        forecasts = []   
        
        # Call the lstmSensorsModelsDetails function to get the details of the LSTM sensors models
        sensors_details = lstmSensorsModelsDetails()
        
        # Get the details of the target sensor from the sensors details
        target_sensor_details = sensors_details[ForecastViewClass.TARGET_SENSOR]

        # Initialize the serializer with the data from the request and set the many attribute to True
        serializedMeasurements = MeasurementsSerializer(
            data=request.data, many=True)

        # Check if the serialized measurements data is valid
        if serializedMeasurements.is_valid():
            # Log that the LSTM data is valid
            print("LSTM data is valid", os.getcwd())

            # Convert the serialized list of measurements to a dictionary
            measurements = serializerListToDict(serializedMeasurements)
            
            # Loop through the forecasting hours
            for hour in range(ForecastViewClass.FORECASTING_HOURS):
                # Construct the file path for the LSTM model
                file_path = os.path.join(
                    settings.BASE_DIR, f"static/lstmModels/{ForecastViewClass.TARGET_SENSOR}//normalized//{hour + 1}H_Forecast//{hour + 1}H_ForecastModel_{target_sensor_details[hour]['bestLag']}_SizeWindow")

                # Call the forecaster function with the target sensor details, measurements, file path, and True flag
                forecast = forecaster(target_sensor_details[hour]['bestLag'], measurements, file_path, True)
                
                # Append the forecast, hour, and error to the forecasts list
                forecasts.append(
                    {
                        'hour': f'Hour {hour + 1}',
                        'LSTM_Forecast': forecast,
                        'error': target_sensor_details[hour]['error']
                    }
                )

            # Return the forecasts as a response with a 200 OK status code
            return Response(forecasts, status=status.HTTP_200_OK)
        else:
            # Log that the LSTM data is invalid
            print("LSTM data is invalid", os.getcwd())
            
            # Return a 400 Bad Request response if the serialized measurements data is not valid
            return Response(status=status.HTTP_400_BAD_REQUEST)

    @api_view(['POST'])
    # @tf.function(reduce_retracing=True)
    def bi_lstmForecastView(request):

        # Initialize an empty list to store the forecasts
        forecasts = []   
        
        # Call the lstmSensorsModelsDetails function to get the details of the LSTM sensors models
        sensors_details = biLstmSensorsModelsDetails()
        
        # Get the details of the target sensor from the sensors details
        target_sensor_details = sensors_details[ForecastViewClass.TARGET_SENSOR]

        serializedMeasurements = MeasurementsSerializer(
            data=request.data, many=True)

        # Check if the serialized measurements data is valid
        if serializedMeasurements.is_valid():
            # Log that the LSTM data is valid

            print("BI LSTM data is valid", os.getcwd())

            # Convert the serialized list of measurements to a dictionary
            measurements = serializerListToDict(serializedMeasurements)
            
            # Loop through the forecasting hours
            for hour in range(ForecastViewClass.FORECASTING_HOURS):
                # Construct the file path for the LSTM model
                file_path = os.path.join(
                    settings.BASE_DIR, f"static/bi_lstmModels/{ForecastViewClass.TARGET_SENSOR}//normalized//{hour + 1}H_Forecast//{hour + 1}H_ForecastModel_{target_sensor_details[hour]['bestLag']}_SizeWindow")


            # Call the forecaster function with the target sensor details, measurements, file path, and True flag
                forecast = forecaster(target_sensor_details[hour]['bestLag'], measurements, file_path, True)              
                
                # Append the forecast, hour, and error to the forecasts list
                forecasts.append(
                                    {
                                        'hour': f'Hour {hour + 1}',
                                        'biLSTM_Forecast': forecast,
                                        'error': target_sensor_details[hour]['error']
                                    }

                                )
                
            # Return the forecasts as a response with a 200 OK status code
            return Response(forecasts, status=status.HTTP_200_OK)
        else:
            # Log that the LSTM data is invalid
            print("LSTM data is invalid", os.getcwd())
            
            # Return a 400 Bad Request response if the serialized measurements data is not valid
            return Response(status=status.HTTP_400_BAD_REQUEST)

def forecaster(lags, serealizedMeasurements, AI_modelFilePath, shouldNormalize):    
    # Transform the serialized measurements into a lagged array format that can be used for forecasting
    laggedMeasurements = forecastDataFormater(serealizedMeasurements, lags)

    # Load the pre-trained AI model
    model = load_model(os.path.join(settings.BASE_DIR, AI_modelFilePath))    
    
    if shouldNormalize:
        # Normalize the lagged measurement data
        laggedMeasurements = normalize_forecast_input_data(laggedMeasurements[0], ForecastViewClass.TARGET_SENSOR_MEANS, ForecastViewClass.TARGET_SENSOR_STANDARD_DEV)
        
        # Make the forecast using the normalized data
        forecast = model.predict(laggedMeasurements)
        # Inverse normalize the forecast
        forecast = inverse_normalize_value(forecast, ForecastViewClass.TARGET_SENSOR_MEANS[1], ForecastViewClass.TARGET_SENSOR_STANDARD_DEV[1])
        
        # Return the forecast
        return forecast[0][0]

    else:
        # Make the forecast without normalizing the data
        forecast = model.predict(laggedMeasurements)

        # Return the forecast
        return forecast[0][0]
    
def forecastDataFormater(serealizedMeasurements, lags):
    # Initialize an array to store the lagged measurement data
    measurementsLagArray = [[]]  
    measureArray = []
    
    # Loop through each lag
    for i in range(lags):  
        # Loop through each measurement
        for measure in serealizedMeasurements[i].values():
            measureArray.append(measure)
            
            

        # Add the lagged measurement data to the array
        measurementsLagArray[0].append(measureArray.copy())
        measureArray.clear()
    
    # Return the lagged measurement array
    return measurementsLagArray

def serializerListToDict(list):
    # Initialize a list to store the serialized data as dictionaries
    measurmentsDict = []

    try:
        # Loop through each serialized data item
        for i in range(len(list.data)):
            # Convert the serialized data item to a dictionary
            measurmentsDict.append(dict(list.data[i]))

        # Return the list of dictionaries
        return measurmentsDict

    except:
        # Print an error message if something went wrong
        print("Something went wrong on serializerListToDict function ")
        
def setTargetForecastingSensorMeansAndStandardDeviation(sensorDataset):
    # This function sets the mean and standard deviation of the target forecasting sensors (T, Rn, H, P, and CO2)
    # using the passed 'sensorDataset' as input
    ForecastViewClass.TARGET_SENSOR_MEANS = [sensorDataset['T'].mean(), sensorDataset['Rn'].mean(), sensorDataset['H'].mean(),
        sensorDataset['P'].mean(), sensorDataset['CO2'].mean()]
    ForecastViewClass.TARGET_SENSOR_STANDARD_DEV = [sensorDataset['T'].std(), sensorDataset['Rn'].std(), sensorDataset['H'].std(),sensorDataset['P'].std(), sensorDataset['CO2'].std()]
    
def normalize_forecast_input_data(data, means, stds):
    # This function normalizes the input data using the means and standard deviations passed as arguments
    normalized_data = []
    for i in range(len(data)):
        normalized_row = [(data[i][j] - means[j]) / stds[j]
                          for j in range(len(data[i]))]
        normalized_data.append(normalized_row)
    return [normalized_data]

def inverse_normalize_value(normalized_value, mean, std):
    # This function performs the inverse normalization operation on the normalized_value using the mean and std
    original_value = normalized_value * std + mean
    return original_value

