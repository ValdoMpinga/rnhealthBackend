from rest_framework import serializers
from .models import User

class User_Serealizer(serializers.Serializer):
   # class Meta:
   #    model  = User
   #    fields = ['first_name', 'last_name', 'email', 'password', 'phone_number']
      
   # def create(self, validated_data):
   #    print(**validated_data)
   #    user = User.objects.create(**validated_data)
   #    return user
      
   first_name = serializers.CharField()
   last_name = serializers.CharField()
   email = serializers.EmailField()
   password = serializers.CharField()
   phone_number = serializers.CharField()   
 