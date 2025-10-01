# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.6
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import jax.numpy as jnp
import jax
from einops import rearrange
from transformers import AutoProcessor, FlaxCLIPModel
from jax import jit, vmap
import numpy as np

# from ASAL: Automating the Search for Artificial Life with Foundation Models [https://sakana.ai/asal]
class CLIP():
    def __init__(self, clip_model="clip-vit-base-patch32"):
        self.processor = AutoProcessor.from_pretrained(f"openai/{clip_model}")
        self.clip_model = FlaxCLIPModel.from_pretrained(f"openai/{clip_model}")

        self.img_mean = jnp.array(self.processor.image_processor.image_mean)
        self.img_std = jnp.array(self.processor.image_processor.image_std)

    def embed_img(self, img):
        """
        img shape (H W C) and values in [0, 1].
        returns shape (D)
        """
        H, W, C = img.shape
        if H!=224 or W!=224:
            img = jax.image.resize(img, (224, 224, C), method='bilinear')
        img = rearrange((img-self.img_mean)/self.img_std, "H W C -> 1 C H W")
        z_img = self.clip_model.get_image_features(img)[0]
        return z_img / jnp.linalg.norm(z_img, axis=-1, keepdims=True)

    def embed_txt(self, prompts):
        """
        prompts is list of strings
        returns shape (B D)
        """
        inputs = self.processor(text=prompts, return_tensors="jax", padding=True)
        z_text = self.clip_model.get_text_features(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])
        return z_text / jnp.linalg.norm(z_text, axis=-1, keepdims=True)


# %%
def point_in_quad(p, quad):
    """
    Test if a point p is inside a convex quadrilateral defined by quad.
    """
    def edge_fn(p0, p1, pt):
        # Returns positive if pt is on the left side of the edge from p0 to p1
        edge = p1 - p0
        to_pt = pt - p0
        return edge[0] * to_pt[1] - edge[1] * to_pt[0]

    signs = jnp.array([
        edge_fn(quad[0], quad[1], p),
        edge_fn(quad[1], quad[2], p),
        edge_fn(quad[2], quad[3], p),
        edge_fn(quad[3], quad[0], p),
    ])
    return jnp.all(signs >= 0) | jnp.all(signs <= 0)

@jit
def render_quads(quads, colors):
    """
    Render a list of convex quadrilaterals to an RGB image.
    
    Args:
        image_size: Tuple of (height, width)
        quads: (N, 4, 2) array of quadrilateral vertices
        colors: (N, 3) array of RGB colors (0.0 to 1.0)
    
    Returns:
        (H, W, 3) RGB image
    """

    yy, xx = jnp.meshgrid(jnp.arange(224, dtype=jnp.int32), jnp.arange(224, dtype=jnp.int32), indexing='ij')
    pixels = jnp.stack([xx, yy], axis=-1).reshape(-1, 2)  # (H*W, 2)

    def rasterize_pixel(p):
        def color_if_inside(quad, color):
            return jax.lax.select(point_in_quad(p, quad), color, jnp.zeros(3, dtype=jnp.int32))
        # Accumulate color contributions (last one wins for overlap)
        pixel_color = jnp.ones(3, dtype=jnp.int32)*255
        for i in range(quads.shape[0]):
            c = color_if_inside(quads[i], colors[i])
            pixel_color = jax.lax.select(jnp.any(c > 0), c, pixel_color)
        return pixel_color

    pixel_colors = vmap(rasterize_pixel)(pixels)
    return pixel_colors.reshape(224, 224, 3)[::-1]

def get_image(Quads):
    empty = [np.array([[250, 250], [250, 250], [250, 250], [250, 250]])]
    n = 25 + 25 - len(Quads["blue"]["quadrilateral"]) - len(Quads["red"]["quadrilateral"])
    quads = jnp.array(empty*n + [np.array([[0., 15], [30., 15], [30., 14.], [0., 14.]])/30*224] + Quads["blue"]["quadrilateral"] + Quads["red"]["quadrilateral"])
    colors = jnp.array([[0,0,0]]*n +[[50, 50, 50]] + Quads["blue"]["colors"] + Quads["red"]["colors"], dtype=jnp.int32)
    return render_quads(quads, colors)
