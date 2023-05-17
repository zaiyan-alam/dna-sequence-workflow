# ACCESS-Pegasus-Examples

This repository contains preconfigured Pegasus Workflows examples including the Pegasus Tutorial, to run on ACCESS resources.
For first time users, we highly recommend to do the notebooks in order, as they build up on concepts in the previous notebooks.

Table of contents
=================

<!--ts-->
   * [Getting started](#getting-started)
   * [Pegasus Tutorial](#pegasus-tutorial)
      * [Introduction API](#introduction-api)
      * [Debugging](#debugging)
      * [Command Line Tools](#command-Line-Tools)
      * [Summary](#summary)
   * [Workflow Examples](#workflow-examples)
      * [Artificial Intelligence](#artificial-intelligence)
         * [Lung Segmentation](#lung-segmentation)
         * [Mask Detection](#mask-Detection)
         * [Orca Sound](#orca-sound)  
      * [Astronomy](#astronomy)
         * [Montage](#montage) 
      * [Bioinformatics](#bioinformatics)
         * [Rosetta](#rosetta)
         * [Alphafold](#alphafold) 
<!--te-->



Getting started
===============

To get started, use a web browser and log in to https://access.pegasus.isi.edu . Use your ACCESS credentials to log in and start a Jupyter notebook
From there, start a shell by clicking the menu (Clusters > Shell), and check out the repostitory:

![Shell Access In Jupyter](./images/terminal-start.png)


```
$ git clone https://github.com/pegasus-isi/ACCESS-Pegasus-Examples.git
```

In Jupyter, navigate to the example you are interested in, and step through the notebook. Once the workflow is submitted, you have to add compute resources with HTCondor Annex.


Pegasus Tutorial
================

Pegasus tutorial notebook, which is intended for new users who want to get a quick overview of Pegasus concepts and usage. This tutorial covers some of the following chapters:

Introduction API
================

This chapter gives a quick overview of Pegasus and it's Python API, along with the information regarding ACCESS resources supported by it.

Debugging
=========
When running complex computations (such as workflows) on complex computing infrastructure (for example HPC clusters), things will go wrong. It is therefore important to understand how to detect and debug issues as they appear. The good news is that Pegasus is doing a good job with the detection part, using for example exit codes, and provides tooling to help you debug. This chapter covers some key points involved in the process of debugging, introducing an error and then detecting it. 


Command Line Tools
==================
Running Pegasus is in a Jupyter notebook is very convenient for tutorials and for smaller workflows, but production workflows are most commonly submitted on dedicated HTCondor submit nodes using command line tools. This chapter of the tutorial uses command line tools for planning, submitting and checking status of a workflow generated in a notebook.


Summary
===============
This chapter covers information regarding how to get in touch with our team, citing Pegasus in your work and contains other useful contact information too.


Workflow Examples
=================
The following examples cover workflows related to various domains as follows:


Artificial Intelligence
=======================
Consists of Machine Learning workflows, with examples showing various steps in a conventional ml-pipeline namely from data-preprocessing to inference.

Lung Segmentation
=================
Precise detection of the borders of organs and lesions in medical images such as X-rays, CT, or MRI scans is an essential step towards correct diagnosis and treatment planning. We implement a workflow that employs supervised learning techniques to locate lungs on X-ray images. Lung instance segmentation workflow uses [Chest X-ray](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4256233/) for predicting lung masks from the images using [U-Net](https://arxiv.org/abs/1505.04597) model.
<img src="/Artificial-Intelligence/LungSegmentation/img/ml_steps.png" style="width: 850px;"/>

Mask Detection
==============
The workflow addresses the problem of determining what percentage of the population is properly wearing masks to better track our collective efforts in preventing the spread of COVID-19 in public spaces. To help solve this problem, it leverages modern deep learning tools such as the Optuna hyper parameter optimization framework and the [FastRCNNPredictor](https://arxiv.org/abs/1506.01497) model. The experiment is organized as a scientific workflow and utilizes the Pegasus Workflow Management System to handle its execution on distributed resources. 
<img src="/Artificial-Intelligence/MaskDetection/imgs/ml_steps3.png" style="width: 900px;"/>

Orca Sound
==========
The [Ocean Observatories Initiative (OOI)](https://oceanobservatories.org/), through a network of sensors, supports critical research in ocean science and marine life. [Orcasound](https://www.orcasound.net/) is a community driven project that leverages hydrophone sensors deployed in **three locations** in the state of **Washington** (San Juan Island, Point Bush, and Port Townsend as shown in the figure below) in order to study Orca whales in the Pacific Northwest region. Throughout the course of this workflow, code to process and analyze the hydrophone data has been developed, and machine learning models have been trained to automatically identify the whistles of the Orcas. All of the code is available publicly on GitHub, and the hydrophone data are free to access, stored in an **AWS S3** bucket. In this paper, we have developed an Orcasound workflow using Pegasus. This version of the pipeline is based on the [Orcasound GitHub actions](https://github.com/orcasound/orca-action-workflow) workflow, and incorporates inference components of the [OrcaHello AI](https://github.com/orcasound/aifororcas-livesystem) notification system.

<img src="/Artificial-Intelligence/OrcaSound/images/ml_steps2.png" style="width: 500px;"/>

Astronomy
=========
The purpose of the examples in this is to illustrate the use of the Pegasus workflows for parallelization of astronomical processing jobs.

Montage
=======
This workflow exhibits a standard flow to the processing of a collection of images to make a mosaic. Using Montage processing tools, we create a mosaic of M17 (the Omega Nebula, 1 degree x 1 degree, in the 2MASS J-band):[From Wikipedia](https://en.wikipedia.org/wiki/Omega_Nebula). The Omega Nebula (M17) is between 5,000 and 6,000 light-years from Earth and it spans some 15 light-years in diameter.

Bioinformatics
==============

Rosetta
=======

Alphafold
=========
