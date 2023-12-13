from django.shortcuts import render
from .models import People, Research, Publication, Project, Asset, FundingSource


def home_view(request):
    # Fetch model objects from database
    people = People.objects.all()
    research_areas = Research.objects.all().order_by('-effort')
    publications = Publication.objects.exclude(title__iexact='Generic')    
    projects = Project.objects.all()
    assets = Asset.objects.all()
    funding_source = FundingSource.objects.all()
    # Pass data to the template
    context = {
        'people': people,
        'research_areas': research_areas,
        'publications': publications,
        'projects': projects,
        'research_areas': research_areas,
        'assets': assets,
        'funding_source': funding_source,

    }
    return render(request, 'home.html', context)

# views.py

