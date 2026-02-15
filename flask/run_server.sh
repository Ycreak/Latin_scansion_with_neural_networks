#!/bin/sh
gunicorn --bind 0.0.0.0:5004 wsgi:app --keyfile /etc/letsencrypt/live/oscc.nolden.biz/privkey.pem --certfile /etc/letsencrypt/live/oscc.nolden.biz/cert.pem
