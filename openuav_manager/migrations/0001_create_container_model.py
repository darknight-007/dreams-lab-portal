# Generated by Django 3.2.25 on 2024-12-25 06:06

from django.db import migrations, models
import uuid


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='Container',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('container_id', models.CharField(max_length=64, unique=True)),
                ('unique_id', models.UUIDField(default=uuid.uuid4, editable=False, unique=True)),
                ('name', models.CharField(max_length=255)),
                ('status', models.CharField(max_length=20)),
                ('created', models.DateTimeField()),
                ('ports', models.JSONField(default=dict)),
                ('image', models.CharField(max_length=255)),
            ],
        ),
    ]
