"""rnhealthBackend URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.0/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path, include
from rest_framework import routers
from forecaster.views import ForecastViewClass
from users.views import UsersViewClass

router = routers.DefaultRouter()

urlpatterns = [
    path('admin/', admin.site.urls),
    
    path('forecast/lstm',ForecastViewClass.lstmForecastView),
    path('forecast/bi-lstm', ForecastViewClass.bi_lstmForecastView),
    path('forecast/target-sensor', ForecastViewClass.targetSensorView),
    path('forecast/normalization', ForecastViewClass.useNormalizationView),
    
    path('user/create', UsersViewClass.createUserView),
    path('user/create/address', UsersViewClass.createUserAddressView)
]

