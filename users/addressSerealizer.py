from rest_framework import serializers

class AddressSerealizer(serializers.Serializer):
   user = serializers.EmailField()
   region = serializers.CharField()
   address = serializers.CharField()
   city = serializers.CharField()
   postal_code = serializers.CharField()
