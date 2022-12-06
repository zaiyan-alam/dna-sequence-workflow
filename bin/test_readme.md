# Alphafold Pegasus Workflow

A Pegasus Workflow for running [Alphafold](https://github.com/deepmind/alphafold) model's inference pipeline regarding protein structure
prediction. The current workflow is regarding the Multiple Sequence Alignment (MSA) and 
Feature Generation steps, which produce a `features.pkl` file that can be later used in protein structure inference
stage using the Alphafold model parameters. The workflow is currently limited to the Alphafold `monomer-system`, thus, there is currently no
options for `model_presets`. 

## Container

The workflow uses a singularity container in order to execute all jobs. It is recommended to build a local container (in a `.sif` file) using the
[Alphafold's](https://github.com/deepmind/alphafold/blob/main/docker/Dockerfile) provided `Dockerfile`. It can be done in the following steps :
```
efef
```
Genetic databases
The workflow does supports both Database presets of Alphafold :
reduced databases--
full databases --



Workflow



Running the workflow


Notes
