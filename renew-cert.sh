#!/bin/bash

# Run certbot with DNS validation
docker-compose run --rm certbot \
  certonly --dns-digitalocean \
  --dns-digitalocean-credentials /etc/letsencrypt/digitalocean.ini \
  --email jnaneshwar.das@gmail.com \
  --agree-tos \
  --no-eff-email \
  -d deepgis.org

# Reload nginx to use new certificate
docker-compose exec nginx nginx -s reload 