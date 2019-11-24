# About
Scripts in this folder is mainly to produce the batch attack results given in the paper. Function of each script is described below:

`baseline_batch_attack.m`: this script produces the results of baseline batch attack (baseline attack schedules the seeds based on loss function values). **For AEC Artifact Evaluation**: the script is configured to directly reproduce the results of Figure 5(a) and Table 9 in the paper, please just execute the `.m` file.

`hybrid_batch_attack.m`: this script produces the results of hybrid batch attack (hybrid attack + two-stage scheduling strategy). **For AEC Artifact Evaluation**: the script is configured to directly reproduce the results of Figure 6(a) and Table 10 in the paper, please just execute the `.m` file. 

`two_stage_strategy.m`: this script produces the result of each stage of the two-stage strategy. **For AEC Artifact Evaluation**: the script is configured to directly reproduce the results of first stage (Figure 2 and Table 7) and second stage (Figure 3 and Table 8) in the paper, please just execute the `.m` file.

`attack_statistics.m`: **For AEC Artifact Evaluation**: this script produces the MNIST and CIFAR10 results in Table 3, Table 4 and Table 5. This script also produces Figure 3(a) in the paper. The transfer rates reported in Table 5 and Table 6 are not included in the results produced by the scripts. However, these transfer rates can be measured directly from the screen outputs of the hybrid attack. When you log the screen output information, the corresponding transfer rates can be easily obtained.  

