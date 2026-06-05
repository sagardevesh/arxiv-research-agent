#!/usr/bin/env bash
# One-time bootstrap for a fresh Ubuntu 24.04 EC2 instance.
#
# Usage (run as root or with sudo):
#   bash setup_ec2.sh
#
# After this script completes:
#   1. Place a .env file at /srv/arxiv-research-agent/.env
#   2. Log in to GHCR (see bottom of script output)
#   3. docker compose -f /srv/arxiv-research-agent/docker-compose.prod.yml up -d
set -euo pipefail

REPO="sagardevesh/arxiv-research-agent"
APP_DIR="/srv/arxiv-research-agent"
BRANCH="main"

echo "=== Installing Docker ==="
apt-get update -q
apt-get install -y -q ca-certificates curl
install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg \
    -o /etc/apt/keyrings/docker.asc
chmod a+r /etc/apt/keyrings/docker.asc
echo \
    "deb [arch=$(dpkg --print-architecture) \
    signed-by=/etc/apt/keyrings/docker.asc] \
    https://download.docker.com/linux/ubuntu \
    $(. /etc/os-release && echo "$VERSION_CODENAME") stable" \
    | tee /etc/apt/sources.list.d/docker.list
apt-get update -q
apt-get install -y -q docker-ce docker-ce-cli containerd.io docker-compose-plugin
usermod -aG docker ubuntu
systemctl enable --now docker

echo "=== Creating app directory ==="
mkdir -p "$APP_DIR"
curl -fsSL \
    "https://raw.githubusercontent.com/$REPO/$BRANCH/infra/docker-compose.prod.yml" \
    -o "$APP_DIR/docker-compose.prod.yml"

echo ""
echo "=== Setup complete ==="
echo ""
echo "Next steps:"
echo "  1. Create $APP_DIR/.env  (copy .env.example from the repo and fill in secrets)"
echo "  2. Log in to GHCR:"
echo "       echo <GHCR_PAT> | docker login ghcr.io -u sagardevesh --password-stdin"
echo "  3. Start services:"
echo "       cd $APP_DIR && docker compose -f docker-compose.prod.yml up -d"
echo "  4. Seed the vector store from your local machine:"
echo "       python scripts/ingest_arxiv.py --query 'transformer attention' --max 50"
