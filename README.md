# Cog SDXL Canny ControlNet with LoRA support

[![Replicate demo and cloud API](https://replicate.com/stability-ai/sdxl/badge)](https://replicate.com/batouresearch/sdxl-controlnet-lora)

This is an implementation of Stability AI's [SDXL](https://github.com/Stability-AI/generative-models) as a [Cog](https://github.com/replicate/cog) model with ControlNet and Replicate's LoRA support.

## Basic Usage

For prediction:

```bash
cog predict -i prompt="shot in the style of sksfer, ..." -i image=@image.png -i lora_weights="https://pbxt.replicate.delivery/mwN3AFyYZyouOB03Uhw8ubKW9rpqMgdtL9zYV9GF2WGDiwbE/trained_model.tar"
```
You may need to increase the default `lora_scale` value to big values such as <strong>0.8 ... 0.95</strong>.

## Limitations
- `lora_weights` only accepts models trained in Replicate and is a mandatory parameter.
- `expert_ensemble_refiner` is currently not supported, you can use `base_image_refiner` instead.
- For the moment, the model only uses canny as the conditional image.

<br>
<strong>For further improvements of this project, feel free to fork and PR!</strong>