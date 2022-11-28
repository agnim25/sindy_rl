#Paper + Appendix 
See file 'AISTATS2021_withappendix.pdf'

# Videos and Experiment
Videos are in 'AISTATSVideos.zip' and experiments were conducted with seperate Matlab-based implementation of the algorithm,

#Source Code
##Requirements
- openAI 'gym'
- stable baselines
- 'cvxpy' + mosek 
- 'control' python package
- the custom ct_cartpole environment

## Installation of Custom Cart-Pole Environment

To install the custom cart-pole gym environment, run the following command in the folder gym-ct_cartpole folder
> 'pip install -e .'

## Running the algorithm on the cartpole swing-up task

Run the following script to evaluate our method and the trained ppo agent on the swing-up task 
> 'eval_script_neurips.py' 

## PPO Training
To plot the learning curve, run
> 'plot_script_neurips.py'

To rerun the ppo training, you need to replace the 'stable_baselines/ppo2/ppo2.py' script with the provided 'ppo2.py' file, and then run 'train_script_ppo2.py'

