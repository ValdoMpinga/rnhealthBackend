from keras.models import load_model
import os
from django.conf import settings
from json import JSONDecoder

def LSTM_forecaster():
    model = load_model(os.path.join(
        settings.BASE_DIR, 'static\lstmModels\D003\\1H_Forecast\\1H_ForecastModel_9_SizeWindow'))
    prediction = model.predict(
        [[
            [33.945, 1328.05, 36, 101.5711667, 59.33333333],
            [33.89833333,	1397.73, 35.83333333, 101.6226667, 58.83333333],
            [33.62666667, 1395.76, 36.33333333,	101.638,	58.5],
            [33.64333333, 1258.793333,	36.33333333,	101.6936667, 59],
            [33.54333333,	1274.03,	36.16666667,	101.715, 58.16666667],
            [33.44166667,	1261.158333, 35.83333333,	101.73, 58.16666667],
            [33.46666667, 1233.271667,	36, 101.7415, 57.33333333],
            [33.36,	1238.586, 36, 101.7366,	57.2],
            [33.4, 1240.485, 36, 101.7616667, 57.33333333],
        ]]
    )

    prediction = ((prediction[0])[0]).item()
    prediction = {'forecast': prediction}
    
    return prediction
