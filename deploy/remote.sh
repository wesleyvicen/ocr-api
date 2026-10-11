#!/usr/bin/env bash
# Roda na VPS (como root), chamado pelo GitHub Actions após enviar o código.
set -euo pipefail

APP_DIR=/var/www/ocr-api
DOMAIN=ocr-api.wesleyv.online
PORT=8011

cd "$APP_DIR"

# 1. Docker
if ! command -v docker >/dev/null 2>&1; then
  echo ">> Instalando Docker"
  curl -fsSL https://get.docker.com | sh
fi

# 2. .env (criado só na primeira vez; edições manuais na VPS são preservadas)
if [ ! -f .env ]; then
  echo ">> Criando .env padrão"
  cp .env.example .env
fi
grep -q '^APP_PORT=' .env || echo "APP_PORT=$PORT" >> .env
PORT=$(grep '^APP_PORT=' .env | cut -d= -f2)

# 3. Build e subida do container
echo ">> Build e start do container"
docker compose -p ocr-api up -d --build --remove-orphans
docker image prune -f >/dev/null

echo ">> Aguardando API em 127.0.0.1:$PORT"
for i in $(seq 1 30); do
  if curl -sf "http://127.0.0.1:$PORT/health" >/dev/null; then
    echo "API ok"
    break
  fi
  if [ "$i" = 30 ]; then
    echo "API não respondeu. Logs:"
    docker compose -p ocr-api logs --tail 50 api
    exit 1
  fi
  sleep 2
done

# 4. nginx (o arquivo só é criado uma vez: o certbot acrescenta o bloco HTTPS nele)
SITE=/etc/nginx/sites-available/$DOMAIN
if [ ! -f "$SITE" ]; then
  echo ">> Configurando nginx para $DOMAIN"
  sed -e "s/__DOMAIN__/$DOMAIN/" -e "s/__PORT__/$PORT/" "$APP_DIR/deploy/nginx-ocr-api.conf" > "$SITE"
  ln -sf "$SITE" "/etc/nginx/sites-enabled/$DOMAIN"
fi
nginx -t
systemctl reload nginx

# 5. HTTPS (Let's Encrypt). Não derruba o deploy se o DNS ainda não propagou.
if [ ! -d "/etc/letsencrypt/live/$DOMAIN" ]; then
  if ! command -v certbot >/dev/null 2>&1; then
    echo ">> Instalando certbot"
    apt-get update -qq && apt-get install -y -qq certbot python3-certbot-nginx
  fi
  echo ">> Emitindo certificado para $DOMAIN"
  certbot --nginx -d "$DOMAIN" --non-interactive --agree-tos --register-unsafely-without-email --redirect \
    || echo "AVISO: certificado não emitido. Confira se o DNS de $DOMAIN aponta para esta VPS e rode o deploy de novo."
fi

echo ">> Deploy concluído: https://$DOMAIN/health"
