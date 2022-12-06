# Alphafold Pegasus Workflow

A Pegasus Workflow for running [Alphafold](https://github.com/deepmind/alphafold) model's inference pipeline regarding protein structure
prediction. The current workflow is regarding the Multiple Sequence Alignment (MSA) and 
Feature Generation steps, which produce a `features.pkl` file that can be later used in protein structure inference
stage using the Alphafold model parameters. The workflow is currently limited to the Alphafold `monomer-system` model preset by default.

## Container

The workflow uses a singularity container in order to execute all jobs. It is recommended to build a local container (in a `.sif` file) using the
[Alphafold's](https://github.com/deepmind/alphafold/blob/main/docker/Dockerfile) provided `Dockerfile` which has all the required libraries and tools. It can be done in the following steps :
```
$ git clone https://github.com/deepmind/alphafold.git
$ cd alphafold
$ docker build -t local/alphafold_container .
$ singularity build alphafold_container.sif docker-daemon://local/alphafold_container
```
The container comes with the following main tools along with other common libraries :
* alphafold==2.2.0
* hmmer==3.3.2 
* hhsuite==3.3.0 
* kalign2==2.04
* absl-py==0.13.0 
* biopython==1.79 
* chex==0.0.7 
* dm-haiku==0.0.4 
* dm-tree==0.1.6 
* immutabledict==2.0.0 


## Genetic databases
If your machine has `aria2c` installed in it, then it's recommended to use Alphafold's provided database download scripts over 
[here](https://github.com/deepmind/alphafold/tree/main/scripts).
Otherwise the database download scripts provided in this repository (`/data/download_all_data.sh`) use readily available command line utilities.
The following databases are used in the workflow :
*   [BFD](https://bfd.mmseqs.com/)
*   [MGnify](https://www.ebi.ac.uk/metagenomics/)
*   [PDB70](http://wwwuser.gwdg.de/~compbiol/data/hhsuite/databases/hhsuite_dbs/)
*   [UniRef90](https://www.uniprot.org/help/uniref)

```
$ /data/download_all_data.sh -d <DOWNLOAD_DIRECTORY>
```
:ledger: **Note:** By default the `download_all_data.sh` script is set to download the reduced version of databases (of size 600 GB). 
If you want to download the full version of databases (of size 2.2 TB), `full_dbs` option can be entered as follows :

```
$ /data/download_all_data.sh -d <DOWNLOAD_DIRECTORY> full_dbs
```

:ledger: **Note: The download directory `<DOWNLOAD_DIR>` should _not_ be a
subdirectory in the AlphaFold repository directory.** If it is, the Docker build
will be slow as the large databases will be copied during the image creation.

## Workflow




The jobs and tools used in the workflow are explained below:

*   `sequence_features` – produces the sequence features from the input fasta file
*   `jackhmmer_uniref90` – runs jackhmmer tool on the UniRef90 database to produce MSAs
*   `jackhmmer_mgnify` – runs jackhmmer tool on the MGnify database to produce MSAs
*   `hhblits_bfd` – runs hhblits tool on the BFD database to produce MSAs
*   `hhsearch_pdb70` - runs hhsearch tool on PDB70 database to produce search templates
*   `msa_features` – turns the MSA results into dicts of features
*   `features_summary` – contains a summary of info reagrding all MSAs produced
*   `combine_features` – combines all MSA features, sequence features and templates into features file `features.pkl`


## Running the workflow

The workflow is set to run on a local HTCondor Pool in the default Condorio
data configuration mode, where each job is run in a Singularity container.
To submit a workflow run :
```
    python3 alphafold_workflow.py \
    --input-fasta-file=/path/to/input/fasta/file \
    --uniref90-db-path=/path/to/uniref90_db \
    --pdb70-db-dir=/path/to/pdb70_db \
    --mgnify-db-path=/path/to/mgnify_db \
    --bfd-db-path=/path/to/bfd_db 
```

