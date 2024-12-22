from django.core.management.base import BaseCommand
from django.db import connection

class Command(BaseCommand):
    help = 'Updates the quiz submission table with new fields'

    def handle(self, *args, **options):
        with connection.cursor() as cursor:
            # Add columns one by one, ignoring errors if they already exist
            new_columns = [
                ('neural_score', 'real NULL DEFAULT 0'),
                ('q11', 'varchar(1) NULL'),
                ('q12', 'varchar(1) NULL'),
                ('q13', 'varchar(1) NULL'),
                ('q14', 'varchar(1) NULL'),
                ('q15', 'varchar(1) NULL'),
            ]

            for column_name, column_type in new_columns:
                try:
                    cursor.execute(f'''
                        ALTER TABLE dreams_laboratory_quizsubmission 
                        ADD COLUMN {column_name} {column_type}
                    ''')
                    self.stdout.write(self.style.SUCCESS(f'Added column {column_name}'))
                except Exception as e:
                    self.stdout.write(self.style.WARNING(f'Column {column_name} might already exist: {str(e)}')) 