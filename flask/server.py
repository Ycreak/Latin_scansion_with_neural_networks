# # # # # #
# OSCC FLASK Server. Handles requests from the OSCC Angular Front end
#      -- As of now is able to handle fragments and users --
#                                                                       #
#                        RUN INSTRUCTIONS                               #
#   docker compose up                                                   #
# # # # # #

# TODO Token authentication between server and front-end
# TODO Input sanitation
# TODO Dont send passwords in plain text to the server

# Add the project root directory (lsnn) to the path
# This allows imports like 'from neural_networks.predict import Predictor' to work.
# We go up one level from the directory containing server.py.
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..") 
# End of path fix

# Library imports
from flask import Flask
from flask_cors import CORS
from flask_restful import Api
import logging

# Class imports
import config as conf

from endpoints.scansion import get_scansion

app = Flask(__name__)
api = Api(app)

# Only allow requests from these specific origins
CORS(app, origins=conf.TRUSTED_ORIGINS)

# Initialize logging
logging.basicConfig(filename=conf.LOG_FILE, level=logging.INFO)

############
# SCANSION #
############
app.add_url_rule("/scansion/get", view_func=get_scansion, methods=["POST"])


# MAIN
if __name__ == "__main__":
    # To run the application, go to the root of the Scansion project and run `python3 flask/server.py`
    app.run(host="0.0.0.0", port=5004, debug=True)
