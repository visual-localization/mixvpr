import numpy as np
import faiss
import faiss.contrib.torch_utils
from miner.frustum_angle_diff import frustum_difference

from prettytable import PrettyTable


def get_validation_recalls(r_list, q_list, k_values,frustum_k_vals, gt, print_results=True, faiss_gpu=False, dataset_name='dataset without name ?',dataset=None,num_references=0):
        
        embed_size = r_list.shape[1]
        if faiss_gpu:
            res = faiss.StandardGpuResources()
            flat_config = faiss.GpuIndexFlatConfig()
            flat_config.useFloat16 = True
            flat_config.device = 0
            faiss_index = faiss.GpuIndexFlatL2(res, embed_size, flat_config)
        # build index
        else:
            faiss_index = faiss.IndexFlatL2(embed_size)
        
        # add references
        faiss_index.add(r_list)

        # search for queries in the index
        _, predictions = faiss_index.search(q_list, max(k_values))
        
        
        # start calculating recall_at_k
        frustum_at_k = np.zeros(frustum_k_vals)
        correct_at_k = np.zeros(len(k_values))
        for q_idx, pred in enumerate(predictions):
            for i, n in enumerate(k_values):
                # if in top N then also in top NN, where NN > N
                filter = np.in1d(pred[:n], gt[q_idx])
                if np.any(filter):
                    if(i<frustum_k_vals):
                        db_idxes = pred[:n][filter]
                        q_idx += num_references
                        max_frustum = max([(frustum_difference(dataset[db_idx],dataset[q_idx]) + frustum_difference(dataset[q_idx],dataset[db_idx]))/2 for db_idx in db_idxes])
                        frustum_at_k[i:] += max_frustum
                    correct_at_k[i:] += 1
                    break
        
        correct_at_k = correct_at_k / len(predictions)
        frustum_at_k = frustum_at_k / len(predictions)
        
        d = {k:v for (k,v) in zip(k_values, correct_at_k)}
        f = {k:v for (k,v) in zip(k_values[:frustum_k_vals], frustum_at_k)}
        if print_results:
            print() # print a new line
            table = PrettyTable()
            table.field_names = ['K']+[str(k) for k in k_values]
            table.add_row(['Recall@K']+ [f'{100*v:.2f}' for v in correct_at_k])
            print(table.get_string(title=f"Performances on {dataset_name}"))
            
            print() # print a new line
            frustum_table = PrettyTable()
            frustum_table.field_names = ['K']+[str(k) for k in k_values[:frustum_k_vals]]
            frustum_table.add_row(['FrustumOverlap@K']+ [f'{100*v:.2f}' for v in frustum_at_k])
            print(frustum_table.get_string(title=f"Performances on {dataset_name}"))
        
        return d,f
