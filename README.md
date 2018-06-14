# GWAVA

## Download source code from sanger ftp

`wget -r ftp://ftp.sanger.ac.uk/pub/resources/software/gwava/v1.0/`

There are different versions of GWAVA v1.0. 
 
## Create conda environment

`conda create -n gwava numpy scipy pandas pytabix scikit-learn=0.14 pybedtools=0.7 matplotlib=2.0.2 libgfortran=1 tabix`

The version number should be defined as above because GWAVA is using old libraries.

`source activate gwava`

`export GWAVA_DIR="${Path_to_your_GWAVA_folder}/v1.0`

## Clone the modified scripts and sorted source data from github

`git clone https://github.com/ding-lab/GWAVA.git`

You need to overite the original files with those from github in the GWAVA folder. The authors are not actively maintained the scripts released in 2014, so GWAVA is only excutable by using those modified scripts.

## Run GWAVA annotation

`python src/gwava_annotate.py test.bed test.csv`

The format of BED file is strickly defined. It should be tab-delimited `chr	start	stop	uniqid`

## Run GWAVA

`python src/gwava.py test.csv out.bed`







