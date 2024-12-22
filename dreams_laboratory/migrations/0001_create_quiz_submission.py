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
    initial = False  # Changed to False since table already exists

    dependencies = [
    ]

    operations = [
        migrations.RunPython(add_new_columns),
    ] 