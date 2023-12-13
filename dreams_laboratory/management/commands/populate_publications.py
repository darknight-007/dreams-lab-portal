import requests
from django.core.management.base import BaseCommand
from dreams_laboratory.models import Publication  # replace 'your_app' with the actual app name

# ... The rest of your imports and command setup

class Command(BaseCommand):
    help = 'Populate publications from DOI'

    def handle(self, *args, **options):
        publications = Publication.objects.exclude(doi__isnull=True).exclude(doi__exact='')

        for publication in publications:
            doi = publication.doi
            url = f'https://api.crossref.org/works/{doi}'

            try:
                response = requests.get(url)
                response.raise_for_status()  # Raises an HTTPError if the HTTP request returned an unsuccessful status code

                data = response.json().get('message')

                # Convert title list to string if necessary
                title = data.get('title')
                if isinstance(title, list):
                    title = title[0] if title else None

                # Parse publication date
                published_date_parts = data.get('published-print', {}).get('date-parts', [[]])
                year, month, day = (published_date_parts[0] + [1, 1])[:3]
                publication_date = f"{year}-{month:02d}-{day:02d}"

                # Update publication fields
                publication.title = title
                publication.authors = ', '.join(
                    [author.get('given') + ' ' + author.get('family') for author in data.get('author', [])]
                )
                publication.publication_date = publication_date
                publication.abstract = data.get('abstract')
                publication.link = data.get('URL')

                # Save the updated publication
                publication.save()
                self.stdout.write(self.style.SUCCESS(f'Successfully updated publication: {doi}'))
            except Exception as e:
                self.stdout.write(self.style.ERROR(f'Failed to fetch data for {doi}: {e}'))
