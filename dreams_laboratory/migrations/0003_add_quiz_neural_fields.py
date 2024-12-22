from django.db import migrations, models

def add_new_columns(apps, schema_editor):
    # Add new columns safely
    try:
        schema_editor.execute('''
            ALTER TABLE "dreams_laboratory_quizsubmission" 
            ADD COLUMN "neural_score" real NULL DEFAULT 0;
        ''')
    except:
        pass  # Column might already exist

    # Add new question columns
    for field in ['q11', 'q12', 'q13', 'q14', 'q15']:
        try:
            schema_editor.execute(f'''
                ALTER TABLE "dreams_laboratory_quizsubmission" 
                ADD COLUMN "{field}" varchar(1) NULL;
            ''')
        except:
            pass  # Column might already exist

class Migration(migrations.Migration):
    dependencies = [
        ('admin', '0001_initial'),  # Keep admin dependency
        ('dreams_laboratory', '0002_update_quiz_submission'),  # Reference previous migration
    ]

    operations = [
        migrations.RunPython(add_new_columns),
    ] 