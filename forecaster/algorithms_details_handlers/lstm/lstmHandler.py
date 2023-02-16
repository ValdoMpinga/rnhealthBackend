from keras.models import load_model
from django.conf import settings

def lstmSensorsModelsDetails():
    details = {
        'non_normalized': 
            {
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
        },
            'normalized': 
                {
                    'D001':
            [
                {
                    'hour': 1,
                    'bestLag': 12,
                    'error': 28.29,

                },
                {
                    'hour': 2,
                    'bestLag': 12,
                    'error': 43.49,
                },
                {
                    'hour': 3,
                    'bestLag': 6,
                    'error': 58.26,
                },
                {
                    'hour': 4,
                    'bestLag': 6,
                    'error': 72.35,
                },
                {
                    'hour': 5,
                    'bestLag': 10,
                    'error': 84.32,
                },
                {
                    'hour': 6,
                    'bestLag': 10,
                    'error': 94.88,
                }],
        'D003':
            [
                {
                    'hour': 1,
                    'bestLag': 6,
                    'error': 52.73,

                },
                {
                    'hour': 2,
                    'bestLag': 6,
                    'error': 98.08,
                },
                {
                    'hour': 3,
                    'bestLag': 9,
                    'error': 135.66,
                },
                {
                    'hour': 4,
                    'bestLag': 11,
                    'error': 163.69,
                },
                {
                    'hour': 5,
                    'bestLag': 11,
                    'error': 192.43,
                },
                {
                    'hour': 6,
                    'bestLag': 11,
                    'error': 225.28,
                }
            ]
                }
    }
    
    return details
