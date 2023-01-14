from rest_framework import serializers


class TargetSensorSerealizer(serializers.Serializer):
   targetSensor = serializers.CharField()
