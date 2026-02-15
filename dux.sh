#!/bin/bash
# Run this script to deploy the application to Dux
rsync -avzuh flask nolden.biz:/home/luukie/servers/oscc/scansion/ --exclude 'flask/env'
rsync -avzuh neural_networks nolden.biz:/home/luukie/servers/oscc/scansion/ --exclude 'neural_networks/.venv'
rsync -avzuh datalake nolden.biz:/home/luukie/servers/oscc/scansion/ --exclude 'datalake/env'
