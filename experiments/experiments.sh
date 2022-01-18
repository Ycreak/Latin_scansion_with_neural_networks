#!/bin/bash
python3 lstm2.py --epochs 25 --exp_hexameter;
python3 lstm2.py --epochs 25 --exp_elegiac;
####
python3 lstm2.py --epochs 25 --exp_train_test --create_model --split 0.2;
python3 lstm2.py --epochs 25 --exp_train_test --create_model --split 0.3;
python3 lstm2.py --epochs 25 --exp_train_test --create_model --split 0.4;
python3 lstm2.py --epochs 25 --exp_train_test --create_model --split 0.5;
python3 lstm2.py --epochs 25 --exp_train_test --create_model --split 0.6;
python3 lstm2.py --epochs 25 --exp_train_test --create_model --split 0.7;
python3 lstm2.py --epochs 25 --exp_train_test --create_model --split 0.8;
python3 lstm2.py --epochs 25 --exp_train_test --create_model --split 0.9;
python3 lstm2.py --epochs 25 --exp_train_test --create_model --split 0.95;