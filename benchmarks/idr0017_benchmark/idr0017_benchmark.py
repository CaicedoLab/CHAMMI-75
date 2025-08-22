import yaml


def run_benchmark(config):
    
    benchmark_test = config['benchmark_test']

    # Gene Compound ROC Test
    if benchmark_test == "auc_roc":

        # Compute Effect Size and Plot the ROC curve
        from tests.gene_compound_roc.gene_compound_roc import GeneCompoundInteraction
        gene_compound_inter = GeneCompoundInteraction(config)
        gene_compound_inter.compute_study_effect_size()
        gene_compound_inter.compute_study_roc()
    


    elif benchmark_test == "replicate_corr":

        # Compute Effect Size and compute the correlation
        from tests.gene_compound_roc.gene_compound_roc import GeneCompoundInteraction
        gene_compound_inter = GeneCompoundInteraction(config)
        gene_compound_inter.compute_study_effect_size()
        gene_compound_inter.compute_replicate_correlation()
        

    # Throw Back Error
    else:
        raise ValueError(f"Unknown benchmark test: {benchmark_test}")

if __name__ == "__main__":

    # Read config.yaml file
    with open("config.yaml", 'r') as f: config = yaml.safe_load(f)

    # Save the config
    

    # Run the benchmark test
    run_benchmark(config)

    



