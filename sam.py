import multiprocessing
from typing import Optional, Tuple

import cv2
import gradio as gr
import numpy as np
import onnxruntime as ort

# OnnxRuntime
sess_options = ort.SessionOptions()
sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
sess_options.intra_op_num_threads = min(8, multiprocessing.cpu_count() // 2)
ort_sess = ort.InferenceSession("decoder_quantized.onnx", sess_options)

# Persistent Args
points_labels = np.array([[1, -1]], dtype=np.float32)
mask_input = np.zeros((1, 1, 256, 256), dtype=np.float32)
has_mask_input = np.zeros(1, dtype=np.float32)


def run_everything(
    image_embedding: np.ndarray,
    attention: np.ndarray,
    image_shape: Tuple[int, int],
    mask_shape: Tuple[int, int],
    border_ratio: float = 0.02,
    n_click_iterations: int = 200,
    initial_object_size: float = 0.2,
    overlap_score_threshold: float = 0.5,
    predicted_iou_threshold: float = 0.8,
    mask_threshold: float = 0.0,
    target_masked_area: float = 0.95,
    eps: float = 1e-7,
    progress=gr.Progress(),
) -> np.ndarray:
    """Find every segments on the image."""
    if image_embedding is None:
        return

    # The output mask size is same as the image_shape
    mask_arr = [np.expand_dims(np.zeros(image_shape), axis=0)]
    # The mask size during operations is mask_shape
    combined_mask = np.zeros(mask_shape, dtype=np.bool_)
    attention_mask = attention == 255

    progress(0, desc="Starting inference")
    for i in progress.tqdm(range(n_click_iterations), desc="Searching for objects"):
        progress(i / n_click_iterations)

        outs = run_a_single_click_iteration(
            image_embedding,
            attention_mask,
            combined_mask,
            i,
            border_ratio,
            n_click_iterations,
            initial_object_size,
            overlap_score_threshold,
            predicted_iou_threshold,
            mask_threshold,
        )

        # Early Stop
        attended_mask = combined_mask & attention_mask
        masked_area = attended_mask.sum() / (attention_mask.sum() + eps)
        if masked_area > target_masked_area:
            break

        # Nothing to update
        if outs is None:
            continue

        mask, pos_mask, neg_mask = outs

        # Combine the masks
        combined_mask |= mask

        # Upsample the mask
        mask = pos_mask * mask + neg_mask
        mask = cv2.resize(mask, dsize=image_shape[::-1])
        mask_arr.append(np.expand_dims(mask > 0, axis=0))
    masks = np.concatenate(mask_arr, axis=0)
    return masks


def run_a_single_click_iteration(
    image_embedding: np.ndarray,
    attention_mask: np.ndarray,
    combined_mask: np.ndarray,
    iteration: int,
    border_ratio: float = 0.02,
    n_click_iterations: int = 200,
    initial_object_size: float = 0.2,
    overlap_score_threshold: float = 0.5,
    predicted_iou_threshold: float = 0.8,
    mask_threshold: float = 0.0,
    eps: float = 1e-7,
) -> Optional[np.ndarray]:
    # Adding borders at edges not to click the area
    inv_mask = (
        np.array(
            (combined_mask == 0) & attention_mask,
            dtype=np.uint8,
        )
        * 255
    )
    offset_x = int(inv_mask.shape[1] * border_ratio)
    offset_y = int(inv_mask.shape[0] * border_ratio)
    inv_mask[:offset_y, :] = inv_mask[:, :offset_x] = 0
    inv_mask[-offset_y:, :] = inv_mask[:, -offset_x:] = 0

    # Get the next click location
    transformed = cv2.distanceTransform(inv_mask, cv2.DIST_L2, 3)
    x, y = get_next_click(transformed, eps)

    return generate_mask(
        image_embedding,
        attention_mask,
        combined_mask,
        x,
        y,
        iteration,
        n_click_iterations,
        initial_object_size,
        overlap_score_threshold,
        predicted_iou_threshold,
        mask_threshold,
        eps,
    )


def generate_mask(
    image_embedding: np.ndarray,
    attention_mask: np.ndarray,
    combined_mask: np.ndarray,
    x: int,
    y: int,
    iteration: int = -1,
    n_click_iterations: int = 0,
    initial_object_size: float = 0.4,
    overlap_score_threshold: float = 0.5,
    predicted_iou_threshold: float = 0.8,
    mask_threshold: float = 0.0,
    eps: float = 1e-7,
) -> Optional[np.ndarray]:
    image_shape = combined_mask.shape[:2]
    mask, predicted_iou = click(image_embedding, x, y, image_shape, mask_threshold)

    # pos and neg masks are necessary for upsampling masks with less loss.
    neg_mask = mask * (mask <= mask_threshold)
    pos_mask = mask * (mask > mask_threshold)
    mask = mask > mask_threshold

    # Filter if not in the attention area.
    overlap_attention = (attention_mask & mask).sum() / mask.sum()
    if 1 - overlap_attention > overlap_score_threshold:
        return None

    # Filter non-confident predictions.
    if predicted_iou[0] < predicted_iou_threshold:
        return None

    # Detect background later
    object_size = mask.sum() / np.prod(mask.shape)
    max_object_size = (iteration + 1) / (n_click_iterations + eps)
    max_object_size *= 1 - initial_object_size
    max_object_size += initial_object_size
    if object_size > max_object_size:
        return None

    # Check they overalps too much
    overlap_score = (mask & combined_mask).sum() / (combined_mask.sum() + eps)
    if overlap_score > overlap_score_threshold:
        return None

    # Remove overlaped area
    mask ^= combined_mask & mask

    # Remove small regions
    filtered_mask = remove_small_regions(mask)
    if filtered_mask is None:
        return None

    mask &= filtered_mask

    return mask, pos_mask, neg_mask


def get_next_click(transformed: np.ndarray, eps: float = 1e-7) -> Tuple[int, int]:
    # Calculate the next click
    # Convert distance transform to a probability map and randomly sample
    transformed_pdf = transformed / (transformed.sum() + eps)
    transformed_cdf = transformed_pdf.flatten().cumsum()
    loc = (np.random.rand(1) < transformed_cdf).argmax()
    y, x = divmod(loc, transformed.shape[1])
    return x, y


def click(
    image_embedding: np.ndarray,
    x: int,
    y: int,
    image_shape: Tuple[int, int],
    mask_threshold: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Click and return the mask."""
    old_h, old_w = image_shape
    new_h, new_w = get_preprocess_shape(old_h, old_w, 1024)
    x *= new_w / old_w
    y *= new_h / old_h

    # Get a regional mask
    points_coords = np.array([[(x, y), (0, 0)]], dtype=np.float32)
    orig_im_size = np.array(image_shape, dtype=np.float32)
    masks, iou_predictions, _ = ort_sess.run(
        None,
        {
            "image_embeddings": image_embedding,
            "point_coords": points_coords,
            "point_labels": points_labels,
            "mask_input": mask_input,
            "has_mask_input": has_mask_input,
            "orig_im_size": orig_im_size,
        },
    )
    return masks[0, 0, :, :], iou_predictions[0]


def get_preprocess_shape(
    oldh: int, oldw: int, long_side_length: int
) -> Tuple[int, int]:
    """
    Compute the output size given input size and target long side length.
    """
    scale = long_side_length * 1.0 / max(oldh, oldw)
    newh, neww = oldh * scale, oldw * scale
    neww = int(neww + 0.5)
    newh = int(newh + 0.5)
    return (newh, neww)


def remove_small_regions(
    mask: np.ndarray, area_thresh: float = 50.0
) -> Optional[np.ndarray]:
    """
    Removes small disconnected regions and holes in a mask. Returns the
    mask and an indicator of if the mask has been modified.
    """
    working_mask = mask.astype(np.uint8) * 255
    contours, _ = cv2.findContours(
        working_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if len(contours) == 0:
        return None

    contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(contour)
    if area < area_thresh:
        return None

    new_mask = np.zeros_like(mask, dtype=np.uint8)
    cv2.drawContours(new_mask, [contour], -1, (255), cv2.FILLED)

    return new_mask > 0
