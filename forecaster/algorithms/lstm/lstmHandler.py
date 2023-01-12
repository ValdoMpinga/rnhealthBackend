from keras.models import load_model
from django.conf import settings

def lstmSensorsModelsDetails():
    details = {
        'D001':
            [
                {
                    'hour': 1,
                    'bestLag': 6,
                    'error': 33.95,

                },
                {
                    'hour': 2,
                    'bestLag': 12,
                    'error': 55.63,
                },
                {
                    'hour': 3,
                    'bestLag': 6,
                    'error': 68.11,
                },
                {
                    'hour': 4,
                    'bestLag': 6,
                    'error': 84.85,
                },
                {
                    'hour': 5,
                    'bestLag': 7,
                    'error': 96.67,
                },
                {
                    'hour': 6,
                    'bestLag': 6,
                    'error': 107.24,
                }],
        'D003':
            [
                {
                    'hour': 1,
                    'bestLag': 10,
                    'error': 52,

                },
                {
                    'hour': 2,
                    'bestLag': 10,
                    'error': 92.58,
                },
                {
                    'hour': 3,
                    'bestLag': 12,
                    'error': 112.21,
                },
                {
                    'hour': 4,
                    'bestLag': 12,
                    'error': 121.82,
                },
                {
                    'hour': 5,
                    'bestLag': 11,
                    'error': 123.75,
                },
                {
                    'hour': 6,
                    'bestLag': 6,
                    'error': 139.27,
                }
            ]
    }
    
    return details
