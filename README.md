# Serif: Typeface generation using diffusion models

This model can do the following:
- Take typographical constraints such as proportions, symmetry, stress
and output embeddings.

- Generate a fully baked typeface given an embedding vector.


## Components

The model consists of the following components:

### Dataset
The datasets are categorized into two families:
1) Full Font Family: All characters are available for this font, allowing
direct full feature extraction. The fonts are stored as TTF files.
2) Partial Font Family: Only a subset of characters are available for this font, allowing
only interpolation of features vs direct extractions. The fonts are stored as PNG files with a metadata file
indicating the identity of each glyph and bounding box.

### Feature Extractor
 This module takes as input a font file (TTF, Woff2 etc.) and outputs features z_font quantifying the font such as:
* x_height_ratio
* cap_height_ratio
* ascender_ratio
* descender_ratio
* vertical_stem
* horizontal_stem
* contrast_ratio
* stress_angle
* avg_width_ratio
* italic_angle
* serifness_score
* ...etc


### Feature Interpolator
This module learns to interpolate features even when a particular glyph is missing.
It can take in an arbitrary set of characters and output a distribution over values of z_font.
This interpolator is used for font families that do not have a full font file but just
a screen shot or image. The interpolator is trained on source full font family data using masked
modeling.

### Embedding Model
This model learns to produce an embedding vector using a VAE, trained with the SDF's as input.

### Diffusion Model
Given a noisy embedding along with the features corresponding to that embedding,this model generates a a denoised signed distance field of the font. It is trained with standard diffusion loss as well as a loss on stem consistency, contrast matching and metric matching.

### Full Architecture
```
Feature Extractor: SDF → z_font 
VAE: SDF → latent
Diffusion: latent + noise → UNet(z_font, glyph_id) → denoise
```