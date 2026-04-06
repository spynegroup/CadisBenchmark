# CadisBenchmark
Benchmarking Causal Discovery (Cadis) Algorithms

## Create the 'cadis' conda environment 
conda create -n cadis
conda activate cadis

// Install pip inside the conda env.  
// Otherwise, when we issue 'pip install <some_pkg>',  
// the pip installed in the base env will get activated and  
// <some_pkg> will be installed in the base env.   
conda install pip  

// gcastle is not available in any anaconda channels  
pip install gcastle  

conda install -c conda-forge pgmpy  

// Export the env details  
conda list --export > cadisList.txt  

## Rebuild the 'cadis' conda environment
conda create --name cadis --file Assets/cadisList.txt