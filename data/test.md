This sample workflow highlights how one can have a hierarchical workflow where,
first compute job generates some input data for a child job which is a sub-workflow
and third compute job takes in the output from the sub-workflow in order to produce
final output.

The jobs and tools used in the workflow are explained below:
*   `curl` – downloads the contents of pegasus.isi.edu homepage in a html file
*   `diamond_subworkflow` – copies the content of downloaded webpage and adds more info to it
*   `wc` – counts the number of lines in the output from the file produced by sub-workflow

File descripotions:
*   `plan.sh` – plans the pegasus workflow (initializes input files, output files, execution etc.)
*   `workflow_generator.py` – it creates the abstract workflow (catalogs regarding inputs, sites and executables)

## Running the workflow
The workflow is set to run on a local HTCondor Pool Condorio
data configuration mode by default. To run the workflow :
```
    ./plan.sh workflow.yml
```
