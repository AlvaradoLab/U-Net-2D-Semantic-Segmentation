Setup instructions in Linux

Install necessary libraries.

```pip install -r requirements.txt```

Look into `train_config2.json` as it acts as the configuration file for the project. 

To run the training, use:

```
python main.py
```

To test a trained model, use:

```
python main.py --test --model <PATH>
```

`training.log` will contain the log for the epochs of training as well as other information. 
