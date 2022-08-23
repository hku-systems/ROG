# ROG Artifact Evaluation
This repository contains codes and materials for artifact evaluation of paper **ROG: A High Performance and Robust Distributed Training System for Robotic IoT** in MICRO 2022.

<!-- vscode-markdown-toc -->
* 1. [Installation](#Installation)
	* 1.1. [Prepared Environments](#PreparedEnvironments)
	* 1.2. [Raw Installation](#RawInstallation)
* 2. [Experiment Workflow](#ExperimentWorkflow)
* 3. [Reproducing Figures](#ReproducingFigures)
	* 3.1.  [Automatically Generated Figures](#AutomaticallyGeneratedFigures)	
	* 3.2. [MicroEvents](#MicroEvents)
	* 3.3. [Customization Settings](#CustomizationSettings)
* 4. [Notes](#Notes)
	* 4.1. [Possible Randomness in Results](#PossibleRandomnessinResults)
	* 4.2. [Dataset](#Dataset)
	* 4.3. [Model](#Model)
	* 4.4. [Details of Our Code](#DetailsofOurCode)
	* 4.5. [Mobility and Instability of Wireless Networks](#MobilityandInstabilityofWirelessNetworks)
* 5. [REFERENCES](#REFERENCES)

<!-- vscode-markdown-toc-config
	numbering=true
	autoSave=true
	/vscode-markdown-toc-config -->
<!-- /vscode-markdown-toc -->


##  1. <a name='Installation'></a>Installation

###  1.1. <a name='PreparedEnvironments'></a>Prepared Environments
Since our artifact evaluation requires configuring many devices, to ease the burden of configuring, we provide access to a well-prepared environment (a cluster with 2 PC, 1 laptop and 2 NVIDIA Jetson Xavier NX and the devices we provide are stationary and being charged) via a ZeroTier network during artifact evaluation.

Please feel free to contact us via micro2022p156@gmail.com to get access to our ZeroTier network.

After joining our ZeroTier network, you can access the devices we used by ssh commands.

###  1.2. <a name='RawInstallation'></a>Raw Installation
Users can also build their own cluster with the files provided in this repository with the following steps:

-   Build the docker image on each device with the dockerfile we provide according to the architecture of your devices, and then we would be running the experiments in the built image.
For X86 devices, run the following commands (replace `Dockerfile_x86-64` with `Dockerfile_arm64` for arm devices):
```
docker build −f Dockerfile_x86−64 −t ROG
docker run −td −−name rog −v /tmp/.X11−unix −−gpus all −e DISPLAY=:0 −−privileged −v /dev:/dev −−network=host −−cap−add=NET_ADMIN −−ipc=host −v "$PWD":/home/work ROG bash
```
-   Enable WiFi hotspot on one of the devices involved and connect all other devices to the hotspot as follows. First, run `sudo hostapd <hotspot configuration file>` to start a hotspot and run `sudo ip a add 10.42.0.1/24 <your WNIC name>` to set the wireless IP address of one device acting as the parameter server. We provide `./scripts/mt7612.conf` as an example for hotspot configuration file and you may modify it at your convenience. The WNIC name can be accessed by running `sudo ip a`. Second, connect all other devices acting as workers to this hotspot via normal WiFi connection tools, like nmtui, and we recommend assigning a static wireless IP address to each device, so that you only need to configure the IPs of workers in `./scripts/run.py` once in the final step.
-   Set the entire repository as an NFS shared folder among all devices for convenient synchronization. Otherwise, the parameter server should copy the code to all worker devices before each experiment, and all worker devices should send all the experiment log records back after each experiment.
-   In the file `./scripts/run.py` on the parameter server, change `hosts`, `wnic_names`, `password` according to the wireless IP addresses, WNIC names, SSH username, and SSH password of your devices. 

##  2. <a name='ExperimentWorkflow'></a>Experiment Workflow
If you have installed and configured the evaluation environments properly (or in the environment we provide), you can start up the experiments by simply executing only one command on the parameter server (the device enabling hotspot):
```
bash run_all.sh
```

`experiment_records.txt` will record all experiments performed, and raw evaluation results will be generated in the `./result/` folder.
For each subfolder in `./result/` folder, the inside folder `./chkpt/` stores the training models checkpointed during the training process and `./log/` folder stores all the experiment log records.  
Then, the accuracies of all checkpointed models will be tested and stored in `accuracy.csv` automatically (see `./scripts/test_checkpoints_on_server.py`).

If the user accidentally terminates the command `bash run_all.sh` or wants to restart the half-run experiments, please delete the result folders of the experiments that have been executed in `./result` folder or comment out the commands that have already been executed in `run_all.sh` to avoid confusion with later experiment results.

##  3. <a name='ReproducingFigures'></a>Reproducing Figures
### 3.1 <a name='AutomaticallyGeneratedFigures'></a>Automatically Generated Figures
At the end of each evaluation item, Figure 1, Figure 9, and Figure 10 in our paper will be drawn **automatically** in the `./figure/` folder via `./scripts/draw_smooth.py` used in `run_all.sh`.
The `./scripts/draw_smooth.py` script takes one parameter: the case of the executed experiment items, like `outdoors` or `indoors`.

If you would like to redraw some of the figures, 
please don't execute `./scripts/draw_smooth.py` separately, because this script will also record the executed experiment record under the same case in `experiment_records.txt`.
Instead, users can redraw the figures via `./scripts/redraw_smooth.py`. 
The specific usage is as follows:
  - Check `experiment_records.txt` to find which case to redraw. 
  - The content of an example `experiment_records.txt` is shown below.
  - Run `python3 scripts/redraw_smooth.py threshold-case0` and figures will be generated in `./figure/` as `threshold-case0-*.pdf` for these cases.
```
#threshold-case0
06-28-02-20-ROG-4-outdoors
06-29-04-41-ROG-20-outdoors
06-29-21-47-ROG-30-outdoors
06-30-17-14-ROG-40-outdoors
```
###  3.2. <a name='MicroEvents'></a>MicroEvents
**We also provide the `./scripts/draw_microevent.py` script for drawing Figure 8 (microevents) in our paper.**
This script takes three parameters: the location of the ROG experiment results to be drawn, rank of the device to be drawn, and the start time.
The rank can only be a positive integer, because the device whose rank is 0 is the parameter server, and we show the microevents of the device for 5 minutes starting from the given start time. Since the training process takes a long time (over 60 minutes), drawing microevents of the whole training process would hide all the details of the figure so we choose to draw a slice of the training process.
An example command using this script is:
```
python3 ./scripts/draw_microevent.py ./result/06-28-02-20-ROG-4-outdoors 1 300
``` 
which will draw the microevents on the device whose rank is 1 starting from the 300th second.
**You may need to adjust the start time and then re-run the script several times to find informative microevents.**


###  3.3. <a name='CustomizationSettings'></a>Customization Settings
To view available settings, run `python3 scripts/run.py --help` and change the parameters of `python3 scripts/run.py  ...` in `run_all.sh`. For example, you can evaluate end-to-end results in the indoor environment (Figure 6) by changing the command line argument of 'outdoors' to 'indoors' in run_all.sh.

Available options are:

```
--library   TEXT      Distributed training system used for ML training. Only three options: BSP, SSP and ROG.
--threshold INTEGER   Threshold for BSP, SSP, and ROG. Especially 0 for BSP.
--control   TEXT      Type of real-world bandwidth replayed. Only two options: outdoors and indoors.
--batchsize INTEGER   The multiple of default batchsize. 
--epoch     INTEGER   Number of epochs to train.
--note      TEXT      Note to mark different evaluation items.
--help      TEXT      Show this message and exit.
```  
-  Since the ROG needs to calculate MTA(a key parameter for ROG) according to the threshold, we only calculate the results whose threshold is less than or equal to 40 for simplicity, so the threshold of the ROG cannot exceed 40.
-  Default batchsizes were set to ensure the same computation time on heterogeneous devices. If all worker devices are homogeneous, please set all elements in the `batch_size` in `./scripts/run.py` to any same integer in line 41.
##  4. <a name='Notes'></a>Notes
###  4.1. <a name='PossibleRandomnessinResults'></a>Possible Randomness in Results
Because we are using momentum to accelerate training (otherwise convergence would take days of time), randomness in training process (especially at the beginning) would be inevitably introduced by the training algorithm and bring a drop in training accuracy at the beginning.
For fair comparison, we recommend running each case multiple times and selected those without such drop at the beginning.
If you want to re-run a specific experiment, such as `06-29-21-47-ROG-30-outdoors`, you need to execute the following command step by step:
- Run `python3 script/run.py ...`, which is the command that generated `06-29-21-47-ROG-30-outdoors` in `run_all.sh`.
- Run `python3 scripts/test_checkpoints_on_server.py` to test the accuracy of the checkpoint models.
You can execute the first step multiple times, and then execute the second step only once.
- Find the new experiment result folder in `result/` folder, like `xx-xx-xx-xx-ROG-30-outdoors`.
- Replace the folder location in `experiment_records.txt` as follows.
- Run `python3 scripts/redraw_smooth.py threshold-case0` to redraw figures.
```
#threshold-case0
06-28-02-20-ROG-4-outdoors
06-29-04-41-ROG-20-outdoors
xx-xx-xx-xx-ROG-30-outdoors
06-30-17-14-ROG-40-outdoors
```

###  4.2. <a name='Dataset'></a>Dataset
The used original Fed-CiFar100 dataset [<sup>1</sup>](#CIFAR-100) [<sup>2</sup>](#cifar100.load_data) is pulled from Tensorflow [<sup>3</sup>](#TensorFlow) and we generated these adversarial examples by adding noise to the original datasets via DeepTest [<sup>4</sup>](#DeepTest).
For simplicity, we provide the dataset with the added noise in the `./datasets/` folder.
###  4.3. <a name='Model'></a>Model
The used ConvMLP[<sup>5</sup>](#ConvMLP-paper) model and its related code are pulled from its official GitHub repository[<sup>6</sup>](#ConvMLP). 
To avoid users downloading the wrong version and wrong folder location, we put the original code of ConvMLP[<sup>6</sup>](#ConvMLP) directly in the `./Convolutional-MLPs/` folder.
###  4.4. <a name='DetailsofOurCode'></a>Details of Our Code
The core functions of ROG are mainly implemented in `ROG_optim.py` and `ROG_utils.py`, while the core functions of our baselines (BSP and SSP) are mainly implemented in `SSP_optim.py` and `SSP_utils.py`.

Users can use ROG to accelerate their own ML training by learning how to call ROG's API from `adapt_noise_ssp.py`.
###  4.5. <a name='MobilityandInstabilityofWirelessNetworks'></a>Mobility and Instability of Wireless Networks
We recommend disabling the mobility of all the devices involved during evaluation. We instead provide `./scripts/limit_bandwidth.sh` and `./scripts/replay_bandwidth.py` scripts based on Traffic Control[<sup>7</sup>](#tc) to replay on each device the real-time bandwidth that we recorded in the identical settings in our evaluation to introduce instability of wireless networks in evaluation.
And it also ensures reproducibility of the results and reliability of the devices since the moving devices can easily run out of energy or crash into obstacles if not being supervised.

##  5. <a name='REFERENCES'></a>REFERENCES

<div id="CIFAR-100"></div>

- [1] [CIFAR-10 and CIFAR-100 datasets](https://www.cs.toronto.edu/~kriz/cifar.html)


<div id="cifar100.load_data"></div>

- [2] [tff.simulation.datasets.cifar100.load_data | TensorFlow federated](https://www.tensorflow.org/federated/api_docs/python/tff/simulation/datasets/cifar100/load_data)

<div id="TensorFlow"></div>

- [3] [Introduction to TensorFlow](https://tensorflow.google.cn/learn)

<div id="DeepTest"></div>

- [4] [Y. Tian, K. Pei, S. Jana, and B. Ray, “Deeptest: Automated testing of deep-neural-network-driven autonomous cars” in Proceedings of the 40th International Conference on Software Engineering, ser. ICSE ’18. New York, NY, USA: Association for Computing Machinery, 2018, p.303–314](https://doi.org/10.1145/3180155.3180220)

<div id="ConvMLP-paper"></div>

- [5] [J. Li, A. Hassani, S. Walton, and H. Shi, “ConvMLP: Hierarchical convolutional MLPs for vision.”](http://arxiv.org/abs/2109.04454)

<div id="ConvMLP"></div>

- [6] [SHI-labs/convolutional-MLPs: [preprint] ConvMLP: Hierarchical convolutional MLPs for vision, 2021](https://github.com/SHI-Labs/Convolutional-MLPs)

<div id="tc"></div>

- [7] [tc(8) - linux manual page](https://man7.org/linux/man-pages/man8/tc.8.html)

