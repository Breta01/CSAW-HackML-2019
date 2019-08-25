# CSAW-HackML-2019

```bash
./Backdoor-Example/
├── data
│   ├── bd_data
│   │   └── bd_test.h5 // this is the test data with backdoor trigger inserted
│   ├── clean_data
│   │   ├── test.h5 // this is clean data used for testing (validation)
│   │   └── train.h5 // this is clean data used for training the network
│   └── sun_glass_trigger.png
├── eval.py // this is the evaluation script
├── gen_backdoor.py // this is script you should modify to generate your backdoored data
└── model
    └── bd_net
        └── bd_net.h5
```

Training/Test data and example code for backdoor trigger generation and network evaluation can be found [here](https://drive.google.com/drive/folders/1Eo_vJK35zWC8yYgGeS9_pw1qFtpn5zeJ?usp=sharing).

## I. Dependencies
   1. Python 3.6.8
   2. Keras 2.2.4
   3. Numpy 1.16.3
   4. Matplotlib 2.2.2
   5. H5py 2.9.0
   6. PIL 6.0.0
   
## II. Generating test backdoored data
   1. In `data/gen_backdoor.py`, change the `poison_data` function and specify the `target_label`.
   2. Execute the python script by running
      `python ./gen_backdoor.py <trigger filename> <clean test data filename>`.
      
      E.g., `python ./gen_backdoor.py ./data/sun_glass_trigger.png ./data/clean_data/test.h5`.
   3. The poisoned data will be stored under `data/bd_data` directory.
   
## III. Evaluating the backdoored model
   1. Store your model in `model/bd_net` directory.
   2. In `eval.py`, change the `data_preprocessing` function with your own preprocessing and execute the script by running
      `python ./eval.py <clean test data filename> <backdoored test data filename> <backdoored model filename>`.
      
      E.g., `python ./eval.py ./data/clean_data/test.h5 ./data/bd_data/bd_test.h5 ./model/bd_net/bd_net.h5`.
   3. Clean data classification accuracy and backdoored data attack success rate will be printed.
