import argparse

def parse_arguments() -> dict:

    # Create the parser
    my_parser = argparse.ArgumentParser(description='My Simple Argument Parser')

    # Add the arguments
    my_parser.add_argument('--hypotactic',
                            metavar='hypotactic',
                            type=str,
                            help='instructions for hypotactic')
    my_parser.add_argument('--pedecerto',
                            metavar='pedecerto',
                            type=str,
                            help='instructions for pedecerto')
    my_parser.add_argument('--clean',
                            metavar='clean',
                            type=str,
                            help='instructions for creating clean datasets from the sources')
    my_parser.add_argument('--dataset',
                            metavar='dataset',
                            type=str,
                            help='instructions for creating a dataset')
    my_parser.add_argument('--lstm',
                            metavar='lstm',
                            type=str,
                            help='instructions for lstm')

    # Execute parse_args()
    args = my_parser.parse_args()

    return args

