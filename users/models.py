from django.contrib.auth.models import AbstractUser
from django.db import models
from django.utils.translation import gettext_lazy as _
from phonenumber_field.modelfields import PhoneNumberField
from .usersManager import UserManager

class User(AbstractUser):
    username = None
    email = models.EmailField(_('email address'), unique=True)
    # first_name = models.CharField(max_length=50, blank=False, null=False)
    # last_name = models.CharField(max_length=50, blank=False, null=False)
    phone_number = PhoneNumberField()

    objects = UserManager()

    USERNAME_FIELD = 'email'
    REQUIRED_FIELDS = ['first_name', 'last_name', 'phone_number']

class Address(models.Model):
    user = models.ForeignKey(User,null=True,on_delete=models.SET_NULL)
    region = models.CharField(max_length=50)
    address = models.CharField(max_length=50)
    city = models.CharField(max_length=50)
    postal_code = models.CharField(max_length=20)
