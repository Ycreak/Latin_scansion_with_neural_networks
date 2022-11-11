#!/bin/bash
python3 lstm.py --custom_train_test --create_model --verbose --epochs 5;
python3 lstm.py --custom_train_test --create_model --verbose --epochs 10;
python3 lstm.py --custom_train_test --create_model --verbose --epochs 15;
python3 lstm.py --custom_train_test --create_model --verbose --epochs 20;
python3 lstm.py --custom_train_test --create_model --verbose --epochs 25;
