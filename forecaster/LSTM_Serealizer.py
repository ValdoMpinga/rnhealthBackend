from rest_framework import serializers


class LSTM_Serealizer(serializers.Serializer):
   temperature = serializers.FloatField()
   radon = serializers.FloatField()
   humidity = serializers.FloatField()
   pressure = serializers.FloatField()
   co2 = serializers.FloatField()
   
 