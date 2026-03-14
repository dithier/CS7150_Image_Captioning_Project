# 1. Make sure you are in virtual environment
```module load cuda/12.1.1 anaconda3/2024.06```
```source activate cs7150```

# 2. Make sure all directories you need exist (logs, runs, and models)

Also, it's a good check to edit train.sh to make sure everything is correct

# 3. Submit request to train
```sbatch train.sh```

# 4. Monitor job as needed
```squeue -u <neu user name>``` checks job status
```tail -f logs/<jobid>.out``` 


# Tensorboard
## Get event file locally
Download the event file from Open On Demand on to your local computer in "runs" folder

## Start tensorboard visualization
In your terminal in the directory that has the runs folder run:
 ```PYTHONWARNINGS="ignore:pkg_resources is deprecated as an API:UserWarning" tensorboard --logdir=runs```


