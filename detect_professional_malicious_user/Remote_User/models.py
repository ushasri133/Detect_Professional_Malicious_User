from django.db import models

# Create your models here.
from django.db.models import CASCADE


class ClientRegister_Model(models.Model):
    username = models.CharField(max_length=30)
    email = models.EmailField(max_length=30)
    password = models.CharField(max_length=10)
    phoneno = models.CharField(max_length=10)
    country = models.CharField(max_length=30)
    state = models.CharField(max_length=30)
    city = models.CharField(max_length=30)
    gender= models.CharField(max_length=30)
    address= models.CharField(max_length=30)

class malacious_user(models.Model):

    Gender= models.CharField(max_length=300)
    Age= models.CharField(max_length=300)
    title= models.CharField(max_length=3000)
    title_orig= models.CharField(max_length=3000)
    units_sold= models.CharField(max_length=300)
    rating= models.CharField(max_length=300)
    tags= models.CharField(max_length=300)
    product_color= models.CharField(max_length=300)
    countries_shipped_to= models.CharField(max_length=300)
    product_url= models.CharField(max_length=30000)
    product_id= models.CharField(max_length=300)
    Prediction= models.CharField(max_length=300)

class detection_accuracy(models.Model):

    names = models.CharField(max_length=300)
    ratio = models.CharField(max_length=300)

class detection_ratio(models.Model):

    names = models.CharField(max_length=300)
    ratio = models.CharField(max_length=300)



