# Generated by Django 3.2.25 on 2024-12-30 05:30

from django.db import migrations, models
import django.utils.timezone


class Migration(migrations.Migration):

    dependencies = [
        ('openuav_manager', '0002_auto_20241225_1705'),
    ]

    operations = [
        migrations.AlterField(
            model_name='container',
            name='created',
            field=models.DateTimeField(default=django.utils.timezone.now),
        ),
        migrations.AlterField(
            model_name='container',
            name='image',
            field=models.CharField(default='openuav:px4-sitl', max_length=255),
        ),
        migrations.AlterField(
            model_name='container',
            name='session_type',
            field=models.CharField(choices=[('guest', 'guest'), ('user', 'user')], default='guest', max_length=10),
        ),
        migrations.AlterField(
            model_name='container',
            name='status',
            field=models.CharField(choices=[('running', 'running'), ('stopped', 'stopped'), ('removed', 'removed'), ('error', 'error')], default='stopped', max_length=20),
        ),
    ]