#!/bin/bash

domains=(deepgis.org)
rsa_key_size=4096
data_path="./certbot"
email="jnaneshwar.das@gmail.com"

# Create dummy certificate
mkdir -p "$data_path/conf/live/$domains"
mkdir -p "$data_path/data"

echo "### Creating initial dummy certificate for $domains ..."
openssl req -x509 -nodes -newkey rsa:$rsa_key_size -days 1\
  -keyout "$data_path/conf/live/$domains/privkey.pem" \
  -out "$data_path/conf/live/$domains/fullchain.pem" \
  -subj "/CN=localhost" || true

echo "### Starting nginx with self-signed certificate..."
docker-compose up --force-recreate -d nginx

echo "Note: You can now set up the DNS TXT record in DigitalOcean for $domains"
echo "The site will work with a self-signed certificate until Let's Encrypt validation is completed"