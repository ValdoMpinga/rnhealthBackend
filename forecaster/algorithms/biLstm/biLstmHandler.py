from keras.models import load_model
from django.conf import settings


def biLstmSensorsModelsDetails():
    details = {
        'D001':
            [
                {
                    'hour': 1,
                    'bestLag': 11,
                    'error': 35.42,

                },
                {
                    'hour': 2,
                    'bestLag': 8,
                    'error': 52.43,
                },
                {
                    'hour': 3,
                    'bestLag': 7,
                    'error': 68.42,
                },
                {
                    'hour': 4,
                    'bestLag': 8,
                    'error': 83.80,
                },
                {
                    'hour': 5,
                    'bestLag': 6,
                    'error': 96.64,
                },
                {
                    'hour': 6,
                    'bestLag': 6,
                    'error': 105.47,
                }],
        'D003':
            [
                {
                    'hour': 1,
                    'bestLag': 8,
                    'error': 50.30,

                },
                {
                    'hour': 2,
                    'bestLag': 9,
                    'error': 90,
                },
                {
                    'hour': 3,
                    'bestLag': 12,
                    'error': 109.87,
                },
                {
                    'hour': 4,
                    'bestLag': 12,
                    'error': 117.22,
                },
                {
                    'hour': 5,
                    'bestLag': 11,
                    'error': 123,
                },
                {
                    'hour': 6,
                    'bestLag': 12,
                    'error': 130.48,
                }
            ]
    }

    return details
