# GAN for Celebrity Face Generation

This project implements a Generative Adversarial Network (GAN) to generate realistic celebrity faces using the [CelebA dataset](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html). The GAN is trained on high-quality images of celebrities and is capable of synthesizing new face images.

---

## Features

- **Custom Data Loader**: Handles the CelebA dataset, including image transformations and loading.
- **GAN Architecture**: Includes both Generator and Discriminator networks.
- **Training Pipeline**: Optimized training with progress tracking and intermediate image generation.
- **Device Support**: Compatible with CPUs, GPUs, and Apple Silicon's MPS (Metal Performance Shaders).

---

## Install requirements

```bash
pip install -r requirements.txt
```

## Dataset

The project uses the CelebA dataset, which contains over 200,000 celebrity images with rich attribute annotations.

### How to Download the Dataset

The dataset is downloaded automatically using the `kagglehub` library. Example:

```python
import kagglehub
path = kagglehub.dataset_download("jessicali9530/celeba-dataset")
```
