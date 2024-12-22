from django.db import migrations, models

class Migration(migrations.Migration):
    dependencies = [
        ('dreams_laboratory', '0001_create_quiz_submission'),
    ]

    operations = [
        migrations.AddField(
            model_name='quizsubmission',
            name='neural_score',
            field=models.FloatField(null=True, default=0),
        ),
        migrations.AddField(
            model_name='quizsubmission',
            name='q11',
            field=models.CharField(max_length=1, null=True, blank=True),
        ),
        migrations.AddField(
            model_name='quizsubmission',
            name='q12',
            field=models.CharField(max_length=1, null=True, blank=True),
        ),
        migrations.AddField(
            model_name='quizsubmission',
            name='q13',
            field=models.CharField(max_length=1, null=True, blank=True),
        ),
        migrations.AddField(
            model_name='quizsubmission',
            name='q14',
            field=models.CharField(max_length=1, null=True, blank=True),
        ),
        migrations.AddField(
            model_name='quizsubmission',
            name='q15',
            field=models.CharField(max_length=1, null=True, blank=True),
        ),
    ] 