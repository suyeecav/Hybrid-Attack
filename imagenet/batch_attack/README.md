# About
Scripts in this folder is mainly to produce the batch attack results given in the paper. Function of each script is described below:

`baseline_batch_attack.m`: this script produces the results of baseline batch attack (baseline attack schedules the seeds based on loss function values). 

`hybrid_batch_attack.m`: this script produces the results of hybrid batch attack (hybrid attack + two-stage scheduling strategy). Specifically, the script is configured to directly reproduce the results of Figure 6(b) and Table 10 in the paper, please just execute the `.m` file. 

`two_stage_strategy.m`: this script produces the result of each stage of the two-stage strategy. 

`attack_statistics.m`: this script produces the ImageNet results in Table 3. It also produces Figure 3(b) in the paper. 

