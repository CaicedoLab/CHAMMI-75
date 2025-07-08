
import pandas as pd
import os
import json
from morphem.evaluation import evaluate, create_umap

import warnings
warnings.filterwarnings("ignore")

def save_results(results, dest_dir, dataset, classifier, knn_metric):
    # Helper function
    # Save results for each dataset as a json dictionary at dest_dir
    full_reports_dict = {}
    full_reports_dict['target_encoding'] = results["encoded_target"]
    for task_ind, task in enumerate(results["tasks"]):
        full_reports_dict[task] = results["reports_dict"][task_ind]

    if not os.path.exists(dest_dir+ '/'):
        os.makedirs(dest_dir+ '/')
    
    if classifier == 'knn':
        dict_path = f'{dest_dir}/{dataset}_{classifier}_{knn_metric}_results.json'
    else:
        dict_path = f'{dest_dir}/{dataset}_{classifier}_results.json'
        
    with open(dict_path, 'w') as f:
        json.dump(full_reports_dict, f)

    return
            
def run_benchmark(root_dir, dest_dir, feature_dir, feature_file, classifier='knn', umap=False, use_gpu=True, knn_metric='l2'):

    # encode dataset, task, and classifier
    task_dict = pd.DataFrame({'dataset':['Allen', 'HPA', 'CP'], 
                              'classifier':[classifier for i in range(3)], \
                              'leave_out': [None, 'Task_three', 'Task_four'], \
                              'leaveout_label': [None, 'cell_type', 'Plate'], \
                              'umap_label': ['Structure', 'cell_type', 'source'] 
                             })
    
    full_result_df = pd.DataFrame(columns=['dataset', 'task', 'classifier', 'accuracy', 'f1_score_macro'])
    
    # Iterrate over each dataset
    for idx, row in task_dict.iterrows():
        dataset        = row.dataset
        classifier     = row.classifier
        leave_out      = row.leave_out
        leaveout_label = row.leaveout_label
        umap_label     = row.umap_label
        
        features_path  = f'{feature_dir}/{dataset}/{feature_file}'
        df_path        = f'{root_dir}/{dataset}/enriched_meta.csv'
        
        # Create umap and run classification
        if umap:
            create_umap(dataset, 
                        features_path, 
                        df_path, 
                        dest_dir, 
                        ['Label', umap_label])
            
        results = evaluate(features_path, 
                           df_path, 
                           leave_out, 
                           leaveout_label, 
                           classifier, 
                           use_gpu, 
                           knn_metric)

        # Print the full results
        print('Results:')
        for task_ind, task in enumerate(results["tasks"]):
            print(f'Results for {dataset} {task} with {classifier} :')
            print(results["reports_str"][task_ind])
        
        # Save results as dictionary
        save_results(results, dest_dir, dataset, classifier, knn_metric)
        
        # Save results as csv
        result_temp = pd.DataFrame({'dataset': [dataset for i in range(len(results["tasks"]))],\
                        'task': results["tasks"],'classifier': [classifier for i in range(len(results["tasks"]))],\
                        'accuracy': results["accuracies"],'f1_score_macro': results["f1scores_macro"]})
        full_result_df = pd.concat([full_result_df, result_temp]).reset_index(drop=True)
    
    if classifier == 'knn':        
        full_result_df.to_csv(f'{dest_dir}/{classifier}_{knn_metric}_full_results.csv', index=False) 
    else:
        full_result_df.to_csv(f'{dest_dir}/{classifier}_full_results.csv', index=False)
        
    return full_result_df
