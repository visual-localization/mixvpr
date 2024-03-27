import torch

from .frustum_angle_diff import FrustumDifferennce, AngleDifference

from const import FRUSTUM_THRESHOLD, ANGLE_THRESHOLD


def convert_batch_to_scenes(scenes):
  res=[]
  for k in scenes:
    for idx,value in enumerate(scenes[k]):
      if(len(res)==idx):
        res.append({})
      res[idx][k] = value
  return res



def custom_get_matches_and_diffs(scenes, labels, ref_scenes=None, ref_labels=None, frustum_overlap_threshold = FRUSTUM_THRESHOLD, angle_threshold = ANGLE_THRESHOLD):
    ref_labels = labels if ref_labels is None else ref_labels
    
    scenes = convert_batch_to_scenes(scenes)
    if ref_scenes is None:
        ref_scenes = scenes
    else:
        ref_scenes = convert_batch_to_scenes(ref_scenes)
    
    frustum_diff = torch.tensor([[FrustumDifferennce.get_frustum_difference(anchor,target) for target in scenes] for anchor in ref_scenes])
    angle_diff = torch.tensor([[AngleDifference.relative_q(anchor["rotation"],target["rotation"]) for target in scenes] for anchor in ref_scenes])
    labels1 = labels.unsqueeze(1)
    labels2 = ref_labels.unsqueeze(0)
    zone_diff = (labels1 == labels2).byte()
    
    ind = torch.triu_indices(frustum_diff.size(dim=0), frustum_diff.size(dim=0), 1)
    frustum_diff[ind[0],ind[1]] = frustum_diff[ind[0],ind[1]] + frustum_diff[ind[1],ind[0]]
    frustum_diff[ind[1],ind[0]] = frustum_diff[ind[0],ind[1]]
    matches = torch.logical_and(frustum_diff >= frustum_overlap_threshold, angle_diff <= angle_threshold,).byte()
    matches = torch.logical_and(matches,zone_diff)
    diffs = matches ^ 1
    if ref_scenes is scenes:
        matches.fill_diagonal_(0)
    return matches, diffs

def custom_get_all_pairs_indices(scenes, labels,ref_scenes = None,ref_labels=None, frustum_overlap_threshold = FRUSTUM_THRESHOLD, angle_threshold = ANGLE_THRESHOLD):
    """
    Given a tensor of labels(list of scenes), this will return 4 tensors.
    The first 2 tensors are the indices which form all positive pairs
    The second 2 tensors are the indices which form all negative pairs
    """
    matches, diffs = custom_get_matches_and_diffs(
      scenes=scenes,ref_scenes=ref_scenes,
      labels=labels, ref_labels=ref_labels,
      frustum_overlap_threshold=frustum_overlap_threshold, 
      angle_threshold=angle_threshold
    )
    a1_idx, p_idx = torch.where(matches)
    a2_idx, n_idx = torch.where(diffs)
    return a1_idx, p_idx, a2_idx, n_idx