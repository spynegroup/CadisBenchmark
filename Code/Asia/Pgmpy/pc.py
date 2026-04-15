'''
Goal: Apply the PC algorithm on the simulated Asia dataset
Author: Saptarshi Pyne
Date: April 14, 2026

'''
#-------------------------------------------------------
## Begin: Import modules 
#-------------------------------------------------------
import os

## loadData.py
from loadData import loadData

from pgmpy.causal_discovery import PC
#-------------------------------------------------------
## End: Import modules 
#-------------------------------------------------------



def main(data, significance_levels, ci_tests, true_model):

    ## Define input parameters
    asia_datafile = os.path.normpath(os.path.join(os.getcwd(), '..', '..', '..', 'Assets', 'Asia', 'asia_dataset.pkl'))

    ## Load the Asia model and data matrix
    asia_model, asia_data = loadData(asia_datafile)

    ## Check whether the model and data matrix are correctly loaded
    print(asia_model)
    print(asia_data.head())


    for alpha, ci_test in product(significance_levels, ci_tests):
        print(f"  PC | alpha={alpha} | ci={ci_test} ...", end=" ", flush=True)
        try:
            dag = PC(data).estimate(
                variant="stable",
                ci_test=ci_test,
                significance_level=alpha,
                return_type="dag",
            )

            PC(ci_test = None, 
	    return_type = 'pdag', 
	    significance_level = 0.01, 
	    max_cond_vars = 5, 
	    expert_knowledge = None, 
	    enforce_expert_knowledge = False, 
	    n_jobs = -1, 
	    show_progress = True)

	    PC.fit(asia_data)

            dag = PC.causal_graph_


            metrics = evaluate_structure(dag, true_model, data, "bic")
            metrics.update({"algorithm": "PC", "alpha": alpha, "ci_test": ci_test})
            records.append(metrics)
            print(f"F1={metrics['f1']:.3f}  SHD={metrics['shd']}")
        except Exception as e:
            print(f"FAILED — {e}")
            records.append({"algorithm": "PC", "alpha": alpha, "ci_test": ci_test, "error": str(e)})
    return records


if __name__ == "__main__":
    main()


























