# Generated by Django 3.2.5 on 2024-01-16 08:56

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('app', '0003_auto_20240116_0741'),
    ]

    operations = [
        migrations.AddField(
            model_name='user',
            name='recent_ten_acc',
            field=models.CharField(default='"ten_acc":[0,0,0,0,0,0,0,0,0,0]', max_length=100),
        ),
    ]
