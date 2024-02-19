import os
import time
from functools import lru_cache

import cv2
import gradio as gr
import numpy as np
import torch
from PIL import Image
from transformers import CLIPSegForImageSegmentation, CLIPSegProcessor

from efficientvit.models.efficientvit.sam import EfficientViTSamPredictor
from efficientvit.sam_model_zoo import create_sam_model
from sam import get_preprocess_shape, run_everything

# Download model weights.
os.system("make model")


IMAGE_SIZE = 1024
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@lru_cache
def load_sam_predictor() -> EfficientViTSamPredictor:
    efficientvit_sam = create_sam_model(name="xl1", weight_url="xl1.pt")
    efficientvit_sam = efficientvit_sam.to(device).eval()
    mask_predictor = EfficientViTSamPredictor(efficientvit_sam)
    return mask_predictor


@lru_cache
def load_clipseg(
    model_name: str = "CIDAS/clipseg-rd64-refined",
) -> tuple[CLIPSegProcessor, CLIPSegForImageSegmentation]:
    processor = CLIPSegProcessor.from_pretrained(model_name)
    model = CLIPSegForImageSegmentation.from_pretrained(model_name).to(device)
    return processor, model


@torch.no_grad()
def get_attention_map(
    image: np.ndarray, query: str, shape: tuple[int, int], attention_threshold: float
) -> np.ndarray:
    processor, model = load_clipseg()
    image = Image.fromarray(image)
    inputs = processor(
        text=query, images=image, padding="max_length", return_tensors="pt"
    ).to(device)
    outputs = model(**inputs)
    attention_map = torch.sigmoid(outputs.logits)
    attention_map = attention_map.detach().cpu().numpy()
    attention_map = (attention_map > attention_threshold).astype(np.uint8) * 255
    attention_map = cv2.resize(attention_map, dsize=shape[::-1])
    return attention_map


def draw_masks(
    image: np.ndarray,
    masks: np.ndarray,
    progress=gr.Progress(),
    alpha: float = 0.4,
) -> np.ndarray:
    b, h, w = masks.shape
    overlay = np.ones((h, w, 3))

    progress(0, desc="Starting drawing")
    for i in progress.tqdm(range(b), desc="Drawing masks"):
        progress(i / b)
        mask, color = masks[i], np.random.random((1, 1, 3))
        overlay = overlay * (1 - mask[..., None]) + color * mask[..., None]
    image = (image * alpha + overlay * 255 * (1 - alpha)).astype(np.uint8)
    return image


def segment(
    image: str,
    query: str,
    n_click_iterations: int,
    mask_size: int,
    attention_threshold: float,
    initial_object_size: float,
    overlap_score_threshold: float,
    predicted_iou_threshold: float,
    masked_area: float,
    progress=gr.Progress(),
) -> tuple[np.ndarray, np.ndarray, str, str]:
    # Read the image and mask.
    image = cv2.imread(image, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Resize the image.
    image_shape = get_preprocess_shape(image.shape[0], image.shape[1], IMAGE_SIZE)
    image = cv2.resize(image, dsize=image_shape[::-1])
    # Get the mask shape
    mask_shape = get_preprocess_shape(image.shape[0], image.shape[1], mask_size)

    # get the image embeddings
    sam_predictor = load_sam_predictor()
    sam_predictor.set_image(image)
    image_embedding = sam_predictor.features.cpu().numpy()
    # image_embedding = image_embedding.reshape(1, 256, 64, 64)

    # Get an attention map
    start = time.perf_counter()
    if query:
        attention = get_attention_map(image, query, mask_shape, attention_threshold)
    else:
        attention = np.ones(mask_shape, dtype=np.uint8) * 255
    masks = run_everything(
        image_embedding,
        attention,
        image_shape,
        mask_shape,
        0.02,  # border ratio
        int(n_click_iterations),
        initial_object_size,
        overlap_score_threshold,
        predicted_iou_threshold,
        0.0,  # mask_threshold
        masked_area,
        1e-7,  # eps
        progress,
    )
    eta = time.perf_counter() - start
    eta_text = f"Searching Time: {eta:.2f} seconds"
    mask_text = f"Found {masks.shape[0] - 1} masks"

    image = draw_masks(image, masks, progress)

    return image, attention, eta_text, mask_text


demo = gr.Interface(
    fn=segment,
    inputs=[
        gr.Image(type="filepath"),
        gr.Text("", label="Object to search"),
        gr.Slider(1, 500, value=200, label="Iteration num. (Larger for more objects)"),
        gr.Slider(256, 1024, value=256, label="Mask size (Larger for more details)"),
        gr.Slider(0, 1, value=0.2, label="Attention threshold (less for less concise)"),
        gr.Slider(0, 1, value=0.2, label="Initial object size to search (Small first)"),
        gr.Slider(0, 1, value=0.5, label="Overlap score bound (Prevent overlaps)"),
        gr.Slider(0, 1, value=0.8, label="IoU prediction threshold (Confidence)"),
        gr.Slider(0, 1, value=0.95, label="Target masked area (Early stop)"),
    ],
    outputs=[
        gr.Image(type="numpy"),
        gr.Image(type="numpy"),
        gr.Label(label="ETA"),
        gr.Label(label="Masks"),
    ],
    allow_flagging="never",
    title="Segment Anything with CLIPSeg",
    examples=[
        [
            os.path.join(os.path.dirname(__file__), "examples/dog.jpg"),
            "dog",
            50,
            256,
            0.6,
            0.2,
            0.5,
            0.8,
            0.95,
        ],
        [
            os.path.join(os.path.dirname(__file__), "examples/city.jpg"),
            "building",
            50,
            256,
            0.6,
            0.2,
            0.5,
            0.8,
            0.95,
        ],
        [
            os.path.join(os.path.dirname(__file__), "examples/food.jpg"),
            "strawberry",
            50,
            256,
            0.6,
            0.2,
            0.5,
            0.8,
            0.95,
        ],
        [
            os.path.join(os.path.dirname(__file__), "examples/horse.jpg"),
            "horse",
            50,
            256,
            0.6,
            0.2,
            0.5,
            0.8,
            0.95,
        ],
        [
            os.path.join(os.path.dirname(__file__), "examples/bears.jpg"),
            "bear",
            50,
            256,
            0.6,
            0.2,
            0.5,
            0.8,
            0.95,
        ],
        [
            os.path.join(os.path.dirname(__file__), "examples/cats.jpg"),
            "cat",
            50,
            256,
            0.6,
            0.2,
            0.5,
            0.8,
            0.95,
        ],
        [
            os.path.join(os.path.dirname(__file__), "examples/fish.jpg"),
            "fish",
            50,
            256,
            0.6,
            0.2,
            0.5,
            0.8,
            0.95,
        ],
    ],
)


load_clipseg()
demo.launch()
