from rest_framework import serializers


class UseNormalizationSerializer(serializers.Serializer):
   useNormalization = serializers.BooleanField()
