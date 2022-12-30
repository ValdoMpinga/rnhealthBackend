from django.shortcuts import render
from .userSerealizer import User_Serealizer
from .addressSerealizer import AddressSerealizer
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
from .models import User, Address
from django.views import View


class UsersViewClass(View):
    @api_view(['POST'])
    def createUserView(request):
        serealizedUserData = User_Serealizer(data=request.data)

        if serealizedUserData.is_valid():
            user = User(
                first_name=serealizedUserData.data['first_name'],
                last_name=serealizedUserData.data['last_name'],
                email=serealizedUserData.data['email'],
                password=serealizedUserData.data['password'],
                phone_number=serealizedUserData.data['phone_number'],
            )
            user.save()

            return Response(serealizedUserData.data, status=status.HTTP_200_OK)
        else:
            return Response(status=status.HTTP_400_BAD_REQUEST)

    @api_view(['POST'])
    def createUserAddressView(request):
        serealizedUserAddressData = AddressSerealizer(data=request.data)

        if serealizedUserAddressData.is_valid():

            userAddress = Address(
                user=User.objects.get(
                    email=serealizedUserAddressData.data['user']),
                region=serealizedUserAddressData.data['region'],
                address=serealizedUserAddressData.data['address'],
                city=serealizedUserAddressData.data['city'],
                postal_code=serealizedUserAddressData.data['postal_code']
            )

            userAddress.save()

            return Response(serealizedUserAddressData.data, status=status.HTTP_200_OK)
        else:
            return Response(status=status.HTTP_400_BAD_REQUEST)
