from django.db import models

# Create your models here.
# 下边类似于数据库的表，每个对象是一行数据，每个属性是一列数据
class User(models.Model):
    username = models.CharField(max_length=20)
    age = models.IntegerField(default=0)
    sex = models.CharField(max_length=10, default='男')
    test_times = models.IntegerField(default=0)
    accuracy_sum = models.FloatField(default=0)
    recent_ten_acc = models.CharField(max_length=100, default='{"ten_acc":[0,0,0,0,0,0,0,0,0,0]}')
    def __str__(self):
        return 'name: '+self.username
    