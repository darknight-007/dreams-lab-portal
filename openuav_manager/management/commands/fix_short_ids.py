from django.core.management.base import BaseCommand
from openuav_manager.models import Container

class Command(BaseCommand):
    help = 'Fix short_id values for all containers'

    def handle(self, *args, **options):
        containers = Container.objects.all()
        for container in containers:
            if container.container_id:
                container.short_id = container.container_id[:12]
            else:
                container.short_id = 'none'
            container.save()
            self.stdout.write(f'Fixed short_id for container {container.name}') 