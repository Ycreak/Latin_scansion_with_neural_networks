import tensorflow as tf
import polars as pl
import numpy as np

def generate_y_pred_true(model: tf.keras.models.Model, test_features: list, test_labels: list) -> tuple[np.ndarray, np.ndarray]:
    """
    Generates a y_prediction and a y_true dataset given the model. Can be used to run a classification report or predict a sentence.
    :param model: Trained Keras model for prediction.
    :param test_features: List of feature arrays for prediction.
    :param test_labels: List of true labels.
    :return: Tuple of predicted labels and true labels.
    """
    # Predict probabilities using the model with multiple input features
    y_pred_probs = model.predict(test_features)

    # Get predicted classes
    y_pred = np.argmax(y_pred_probs, axis=-1)

    # Flatten the true labels and predicted labels
    y_true_flat = test_labels.flatten()
    y_pred_flat = y_pred.flatten()

    return y_pred_flat, y_true_flat

def get_candidate_meters_from_df(df: pl.DataFrame, minimum_number_of_lines: int) -> list[dict]:
    """
    Returns a list with candidate meters given a specific cutoff. For example, if we need 1000 lines of 
    poetry per meter, we can find the candidate meters in our dataframe with this function.
    """
    # Find the number of lines we have per meter type
    meters_with_counts = (
        df.group_by("meter")
        .agg(pl.col("line_number").n_unique().alias("number_of_lines"))
        .sort("number_of_lines", descending=True)
    )

    # Filter dataframe where we have at least 1000 lines
    return [{'meter': d['meter'], 'lines': d['number_of_lines']} for d in meters_with_counts.to_dicts() if d["number_of_lines"] > minimum_number_of_lines]

