from rest_framework import serializers


class HoursSerealizer(serializers.Serializer):
   hour1 = serializers.BooleanField()
   hour2 = serializers.BooleanField()
   hour3 = serializers.BooleanField()
   hour4 = serializers.BooleanField()
   hour5 = serializers.BooleanField()
   hour6 = serializers.BooleanField()
