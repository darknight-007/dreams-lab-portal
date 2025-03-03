import requests
from bs4 import BeautifulSoup
from django.core.management.base import BaseCommand
from dreams_laboratory.models import Publication

class Command(BaseCommand):

    def handle(self, *args, **options):
        # Replace 'your_google_scholar_id' with your actual Google Scholar ID
        url = 'https://scholar.google.com/citations?user=Eja4Kw4AAAAJ&hl=en&oi=ao'
        
        # Send request to Google Scholar
        page = requests.get(url)
        soup = BeautifulSoup(page.text, 'html.parser')
        print(soup)
        # Find all DOIs on the page, assuming they are contained in 'a' elements with class 'doi'
        dois = [a.text for a in soup.find_all('a', class_='doi')]

        for doi in dois:
          print(doi)
            # Check if publication with DOI already exists
          if not Publication.objects.filter(doi=doi).exists():
              # If not, insert the new publication into the database
              Publication.objects.create(doi=doi)
