#!/bin/bash
# You have to install all requirements first
# pip install -r requirements.txt

# Generate the data
echo "Generating data"
python ./gen_backdoor.py x ./data/clean_data/test.h5

# Run the test on Plain network
echo "Evaluting Plain network"
python ./eval.py ./data/clean_data/test.h5 ./data/bd_data/bd_test.h5 ./model/bd_net/network_plain.h5

# Run the test on Generator network
echo "Evaluating Generator network"
python ./eval_gen.py ./data/clean_data/test.h5 ./data/bd_data/bd_test.h5 ./model/bd_net/network_generator.h5
