#!/bin/bash
#
# Script for running `streamlit` safely online.
#
#   Usage: PYTHONPATH=. bash ./run/utils/streamlit-run.sh <path_to_streamlit app>
#
# You may need adjust the firewall to allow for remote connections first, like so:
#
#   gcloud compute instances add-tags $VM_NAME --zone $VM_ZONE --tags http-server
#

EXTERNAL_IP=$(curl -s http://metadata.google.internal/computeMetadata/v1/instance/network-interfaces/0/access-configs/0/external-ip -H "Metadata-Flavor: Google")
USER=$(hostname)
PASSWORD=$(openssl rand -hex 16) # Generate a random password for basic authentication
HTPASSWD=/etc/nginx/.htpasswd
CONFIG=/etc/nginx/conf.d/streamlit.conf
PORT=8501 # Default Streamlit port
INFO="\e[36m[INFO]\e[0m"
SUCCESS="\e[32m[SUCCESS]\e[0m"

echo -e "$INFO Installing Nginx..."
sudo apt-get update
sudo apt-get install -y nginx apache2-utils

echo -e "$INFO Setting up basic authentication for Nginx..."
sudo htpasswd -bc $HTPASSWD $USER $PASSWORD

echo -e "$INFO Creating an Nginx configuration file for the Streamlit app..."
sudo tee $CONFIG <<EOF
server {
    listen 80 default_server;
    listen [::]:80 default_server;
    server_name _;

    location / {
        auth_basic "Restricted";
        auth_basic_user_file $HTPASSWD;
        proxy_pass http://localhost:$PORT;
        proxy_http_version 1.1;
        proxy_set_header Host \$host;
        proxy_set_header Upgrade \$http_upgrade;
        proxy_set_header Connection "upgrade";
    }
}
EOF

echo -e "$INFO Removing the default Nginx configuration..."
sudo rm /etc/nginx/sites-enabled/default

echo -e "$INFO Testing the Nginx configuration..."
sudo nginx -t || exit 1

echo -e "$INFO Reloading Nginx to apply the new configuration..."
sudo systemctl reload nginx

echo -e "$SUCCESS You can now view your Streamlit app online."
echo -e "$SUCCESS http://$USER:$PASSWORD@$EXTERNAL_IP/"

# Launch the Streamlit app
streamlit run "$@" --server.port $PORT &

# Function to clean up after the Streamlit app is exited
cleanup() {
  echo -e "$INFO Cleaning up Nginx configuration and password file..."
  sudo rm $CONFIG
  sudo rm $HTPASSWD
  sudo systemctl restart nginx
}

# Trap Ctrl-C to clean up before exiting
trap cleanup INT

# Wait for the Streamlit app to finish
wait
