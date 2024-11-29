import numpy as np
import torch

def convert_ltwh_to_ltrb(bboxes_ltwh):
    """
    Convert bounding boxes from LTWH format to LTRB format.

    Args:
    - bboxes_ltwh (numpy.ndarray or torch.Tensor): Bounding boxes in LTWH format of shape (B, N, 4),
                                  where B is the batch size, N is the number of bounding boxes,
                                  and the last dimension represents (left, top, width, height).

    Returns:
    - numpy.ndarray or torch.Tensor: Bounding boxes in LTRB format of shape (B, N, 4),
                    where the last dimension represents (left, top, right, bottom).
    """
    if isinstance(bboxes_ltwh, np.ndarray):
        # Convert NumPy array to PyTorch tensor
        bboxes_ltwh_tensor = torch.tensor(bboxes_ltwh, dtype=torch.float32)
    elif isinstance(bboxes_ltwh, torch.Tensor):
        bboxes_ltwh_tensor = bboxes_ltwh
    else:
        raise ValueError("Unsupported input type. Input should be a NumPy array or a PyTorch tensor.")

    # Extract the left, top, width, and height components
    left, top, width, height = torch.split(bboxes_ltwh_tensor, 1, dim=-1)

    # Calculate right and bottom components
    right = left + width
    bottom = top + height

    # Concatenate the components to form LTRB bounding boxes
    bboxes_ltrb_tensor = torch.cat((left, top, right, bottom), dim=-1)

    # Convert the result back to NumPy array if the input was a NumPy array
    if isinstance(bboxes_ltwh, np.ndarray):
        bboxes_ltrb_numpy = bboxes_ltrb_tensor.numpy()
        return bboxes_ltrb_numpy
    else:
        return bboxes_ltrb_tensor


def bbox_ltwh2ltrb(ltwh):
    return np.concatenate((ltwh[:2], ltwh[:2] + ltwh[2:]))


def unnormalize_bbox(bbox, image_shape):
    return bbox * (list(image_shape) * 2)


def normalize_bbox(bbox, image_shape):
    return bbox / image_shape.repeat(2)


def normalize_kps(kps, image_shape):
    nm_kps = kps.clone()
    nm_kps[..., :2] = kps[..., :2] / image_shape
    return nm_kps

def center_bboxes_keypoints(batch):
    def normalize_bbox(bbox, W, H):
        bbox[..., 0] = (bbox[..., 0] - W[..., 0]/2) / W[..., 0]
        bbox[..., 1] = (bbox[..., 1] - H[..., 0]/2) / H[..., 0]
        bbox[..., 2] = bbox[..., 2] / W[..., 0]
        bbox[..., 3] = bbox[..., 3] / H[..., 0]
        return bbox

    def normalize_keypoints(kps, W, H):
        W = W.unsqueeze(-1)
        H = H.unsqueeze(-1)
        kps[..., 0] = (kps[..., 0] - W[..., 0]/2) / W[..., 0]
        kps[..., 1] = (kps[..., 1] - H[..., 0]/2) / H[..., 0]
        return kps

    if "det_feats" in batch:
        if "bbox_ltwh" in batch["det_feats"]:
            batch["det_feats"]["bbox_ltwh"] = normalize_bbox(batch["det_feats"]["bbox_ltwh"],
                                                             batch["det_feats"]["im_width"],
                                                             batch["det_feats"]["im_height"])
        if "keypoints_xyc" in batch["det_feats"]:
            batch["det_feats"]["keypoints_xyc"] = normalize_keypoints(batch["det_feats"]["keypoints_xyc"],
                                                                      batch["det_feats"]["im_width"],
                                                                      batch["det_feats"]["im_height"]
                                                                      )
    if "track_feats" in batch:
        if "bbox_ltwh" in batch["det_feats"]:
            batch["track_feats"]["bbox_ltwh"] = normalize_bbox(batch["track_feats"]["bbox_ltwh"],
                                                               batch["track_feats"]["im_width"],
                                                               batch["track_feats"]["im_height"]
                                                               )
        if "keypoints_xyc" in batch["det_feats"]:
            batch["track_feats"]["keypoints_xyc"] = normalize_keypoints(batch["track_feats"]["keypoints_xyc"],
                                                                        batch["track_feats"]["im_width"],
                                                                        batch["track_feats"]["im_height"]
                                                                        )
    return batch

def positive_bboxes_keypoints(batch):
    def normalize_bbox(bbox, W, H):
        bbox[..., 0] = bbox[..., 0] / W[..., 0]
        bbox[..., 1] = bbox[..., 1] / H[..., 0]
        bbox[..., 2] = bbox[..., 2] / W[..., 0]
        bbox[..., 3] = bbox[..., 3] / H[..., 0]
        return bbox

    def normalize_keypoints(kps, W, H):
        kps[..., 0] = kps[..., 0] / W.unsqueeze(-1)[..., 0]
        kps[..., 1] = kps[..., 1] / H.unsqueeze(-1)[..., 0]
        return kps

    if "det_feats" in batch:
        if "bbox_ltwh" in batch["det_feats"]:
            batch["det_feats"]["bbox_ltwh"] = normalize_bbox(batch["det_feats"]["bbox_ltwh"],
                                                             batch["det_feats"]["im_width"],
                                                             batch["det_feats"]["im_height"])
        if "keypoints_xyc" in batch["det_feats"]:
            batch["det_feats"]["keypoints_xyc"] = normalize_keypoints(batch["det_feats"]["keypoints_xyc"],
                                                                      batch["det_feats"]["im_width"],
                                                                      batch["det_feats"]["im_height"]
                                                                      )
    if "track_feats" in batch:
        if "bbox_ltwh" in batch["det_feats"]:
            batch["track_feats"]["bbox_ltwh"] = normalize_bbox(batch["track_feats"]["bbox_ltwh"],
                                                              batch["track_feats"]["im_width"],
                                                              batch["track_feats"]["im_height"]
                                                              )
        if "keypoints_xyc" in batch["det_feats"]:
            batch["track_feats"]["keypoints_xyc"] = normalize_keypoints(batch["track_feats"]["keypoints_xyc"],
                                                                      batch["track_feats"]["im_width"],
                                                                      batch["track_feats"]["im_height"]
                                                                      )
    return batch


norm_coords_strats = {
    "centered": center_bboxes_keypoints,
    "positive": positive_bboxes_keypoints,
}