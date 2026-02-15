"""
This endpoint handles all document types. At the moment, we support the following:
    introductions
    fragments
    testimonia
    playgrounds
These are differentiated via the 'type' parameter. This endpoint checks the incoming
document type and redirects the document to the correct endpoint.
"""

import logging
from flask import request, make_response
from flask_jsonpify import jsonify
from mqdq.cltk_hax.syllabifier import Syllabifier
import polars as pl

from neural_networks.predict import Predictor

def get_scansion() -> object:
    """Reads the cache file and returns it. Needs a sandbox as parameter."""
    try:
        scansion: str = request.get_json()["scansion"]
        words: list[str] = scansion.split()
        S = Syllabifier()
        syllables_per_word: list[str] = []
        for word in words:
            # Also append after each word the space sign
            syllables_per_word.append(S.syllabify(word) + ['-'])

        # 2. Create the initial DataFrame
        # We combine the two lists into a Polars Series and then a DataFrame
        df = pl.DataFrame({
            "word": words,
            "syllables": syllables_per_word
        })

        # 3. Explode the 'syllables' column
        # This converts the list of lists into individual rows while duplicating the 'word'
        result_df = df.explode("syllables").rename({"syllables": "syllable"})

        result_df = result_df.with_columns(
            pl.when(pl.col("syllable") == "-")
              .then(pl.lit("space"))
              .otherwise(pl.col("word"))
              .alias("word") # Overwrite the existing 'word' column
        )

        # Remove the last superfluous space
        result_df = result_df[:-1] 

        # Add the other columns of the dataframe
        result_df = result_df.with_columns(
            pl.lit("Ovidius").alias("author"),
            pl.lit("predict").alias("meter"),
            pl.lit(1).cast(pl.Int64).alias("line_number"), # Cast to an appropriate integer type
            pl.lit("long").alias("label"),
        )

        # Now we have a dataframe just like the one in our datalake. We can give this one to our predictor!
        predictor = Predictor()
        prediction = predictor.run(result_df)

        print('Prediction is done')

        return jsonify(prediction)
    except KeyError as e:
        logging.error(e)
        return make_response("No scansion received.", 422)


