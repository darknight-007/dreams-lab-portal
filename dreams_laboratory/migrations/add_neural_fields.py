from django.db import migrations

class Migration(migrations.Migration):
    dependencies = []  # No dependencies

    operations = [
        migrations.RunSQL(
            # SQL to add new columns
            sql='''
                ALTER TABLE dreams_laboratory_quizsubmission 
                ADD COLUMN neural_score real NULL DEFAULT 0;
                
                ALTER TABLE dreams_laboratory_quizsubmission 
                ADD COLUMN q11 varchar(1) NULL;
                
                ALTER TABLE dreams_laboratory_quizsubmission 
                ADD COLUMN q12 varchar(1) NULL;
                
                ALTER TABLE dreams_laboratory_quizsubmission 
                ADD COLUMN q13 varchar(1) NULL;
                
                ALTER TABLE dreams_laboratory_quizsubmission 
                ADD COLUMN q14 varchar(1) NULL;
                
                ALTER TABLE dreams_laboratory_quizsubmission 
                ADD COLUMN q15 varchar(1) NULL;
            ''',
            # Reverse SQL (if needed)
            reverse_sql='''
                ALTER TABLE dreams_laboratory_quizsubmission 
                DROP COLUMN neural_score;
                
                ALTER TABLE dreams_laboratory_quizsubmission 
                DROP COLUMN q11;
                
                ALTER TABLE dreams_laboratory_quizsubmission 
                DROP COLUMN q12;
                
                ALTER TABLE dreams_laboratory_quizsubmission 
                DROP COLUMN q13;
                
                ALTER TABLE dreams_laboratory_quizsubmission 
                DROP COLUMN q14;
                
                ALTER TABLE dreams_laboratory_quizsubmission 
                DROP COLUMN q15;
            '''
        )
    ] 