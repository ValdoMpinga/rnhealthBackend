from rest_framework import serializers


class MeasurementsSerializer(serializers.Serializer):
   T = serializers.FloatField()
   Rn = serializers.FloatField()
   H = serializers.FloatField()
   P = serializers.FloatField()
   CO2 = serializers.FloatField()
   
 