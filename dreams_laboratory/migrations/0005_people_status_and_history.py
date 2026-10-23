from django.db import migrations, models


def mark_alumni(apps, schema_editor):
    People = apps.get_model('dreams_laboratory', 'People')
    People.objects.filter(role__position__iexact='alumni').update(status='alumni')


class Migration(migrations.Migration):

    dependencies = [
        ('dreams_laboratory', '0004_auto_20251126_1811'),
    ]

    operations = [
        migrations.AlterModelOptions(
            name='people',
            options={'verbose_name_plural': 'people'},
        ),
        migrations.AlterField(
            model_name='people',
            name='email',
            field=models.EmailField(blank=True, max_length=254),
        ),
        migrations.AlterField(
            model_name='people',
            name='profile_pic',
            field=models.URLField(blank=True, max_length=300),
        ),
        migrations.AddField(
            model_name='people',
            name='status',
            field=models.CharField(
                choices=[('current', 'Current member'), ('alumni', 'Alumni'), ('unlisted', 'Not listed')],
                db_index=True, default='current', max_length=20),
        ),
        migrations.AddField(
            model_name='people',
            name='graduation_year',
            field=models.PositiveSmallIntegerField(
                blank=True, null=True, help_text='Expected or actual graduation year.'),
        ),
        migrations.AddField(
            model_name='people',
            name='focus',
            field=models.CharField(
                blank=True, max_length=200,
                help_text='Research focus, e.g. "Radio astronomy · smart sensing".'),
        ),
        migrations.AddField(
            model_name='people',
            name='next_position',
            field=models.CharField(
                blank=True, max_length=200,
                help_text='Where an alumnus went next, e.g. "Faculty, University of Utah".'),
        ),
        migrations.AddField(
            model_name='role',
            name='display_order',
            field=models.PositiveSmallIntegerField(
                default=100, help_text='Lower values list first on the People page.'),
        ),
        migrations.RunPython(mark_alumni, migrations.RunPython.noop),
    ]
