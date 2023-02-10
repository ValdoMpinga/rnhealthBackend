from rest_framework import serializers


class TargetSensorSerializer(serializers.Serializer):
   targetSensor = serializers.CharField()
