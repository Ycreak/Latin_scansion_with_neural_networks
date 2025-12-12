"""

       _   _ _ _ _   _
      | | (_) (_) | (_)
 _   _| |_ _| |_| |_ _  ___  ___   _ __  _   _
| | | | __| | | | __| |/ _ \/ __| | '_ \| | | |
| |_| | |_| | | | |_| |  __/\__ \_| |_) | |_| |
 \__,_|\__|_|_|_|\__|_|\___||___(_) .__/ \__, |
                                  | |     __/ |
                                  |_|    |___/

"""

import hashlib
import binascii
import os
import string
from fuzzywuzzy import fuzz
import json

import config as conf


def verify_password(stored_pwd, provided_pwd) -> bool:
    salt = stored_pwd[:64]
    stored_pwd = stored_pwd[64:]
    pwd_hash = hashlib.pbkdf2_hmac(
        conf.PWD_HASH_NAME,
        provided_pwd.encode("utf-8"),
        salt.encode("ascii"),
        conf.PWD_HASH_ITERATIONS,
    )
    pwd_hash = binascii.hexlify(pwd_hash).decode("ascii")
    return pwd_hash == stored_pwd


def hash_password(password) -> str:
    salt = hashlib.sha256(os.urandom(60)).hexdigest().encode("ascii")
    pwd_hash = hashlib.pbkdf2_hmac(
        conf.PWD_HASH_NAME, password.encode("utf-8"), salt, conf.PWD_HASH_ITERATIONS
    )
    pwd_hash = binascii.hexlify(pwd_hash)
    return (salt + pwd_hash).decode("ascii")


def similarity(a, b):
    a = a.translate(str.maketrans("", "", string.punctuation)).lower()
    b = b.translate(str.maketrans("", "", string.punctuation)).lower()
    return fuzz.token_sort_ratio(a, b)


def write_json(given_object: object, file: str) -> None:
    """Writes the given object to the cache

    Args:
        object (object): to store as json
        file (str): path + filename
    """
    # Serializing json
    json_object = json.dumps(given_object, indent=2)
    # Writing to sample.json
    with open(file, "w") as outfile:
        outfile.write(json_object)


def read_json(file: str) -> list:
    """Reads the requested object from the cache

    Args:
        file (str): path + filename
    """
    with open(file, "r") as openfile:
        # Reading from json file
        return json.load(openfile)
