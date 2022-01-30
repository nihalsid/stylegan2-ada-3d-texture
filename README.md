# stylegan2-ada-3d-texture

Surface texture GAN.
Configuration provided with hydra config file `config/stylegan2.yaml`. Once configured, train with:

## Dataset

```bash
create symlinks
wget xyz
unzip xyz
```

## Output Directories

```bash
symlinks
```

## Running Experiments

```bash
python trainer/train_stylegan.py wandb_main=True
```

## Configuration

Configuration can be overriden with command line flags.

| Key | Description | Default |
| ----|-------------|---------|
|`dataset_path`| Directory with processed data||
|`mesh_path`| Directory with processed mesh (highest res)||
|`pairmeta_path`| Directory with metadata for image-shape pairs (photoshape specific)||
|`df_path`| not used anymore ||
|`image_path`| real images ||
|`mask_path`| real image segmentation masks ||
|`condition_path`| not used anymore ||
|`stat_path`| not used anymore ||
|`uv_path`| processed uv data (for uv baseline) ||
|`silhoutte_path`| texture atlas silhoutte data (for uv baseline) ||
|`mesh_resolution`| not used anymore||
|`experiment`| Experiment name used for logs |`fast_dev`|
|`wandb_main`| If false, results logged to "<project>-dev" wandb project (for dev logs)|`False`|
|`num_mapping_layers`| Number of layers in the mapping network |2|
|`lr_g`| Generator learning rate | 0.002|
|`lr_d`| Discriminator learning rate |0.00235|
|`lr_e`| Encoder learning rate |0.0001|
|`lambda_gp`| Gradient penalty weight | 0.0256 |
|`lambda_plp`| Path length penalty weight |2|
|`lazy_gradient_penalty_interval`| Gradient penalty regularizer interval |16|
|`lazy_path_penalty_after`| Iteration after which path lenght penalty is active |0|
|`lazy_path_penalty_interval`| Path length penalty regularizer interval |4|
|`latent_dim`| Latent dim of starting noise and mapping network output |512|
|`image_size`| Size of generated images |64|
|`num_eval_images`| Number of images on which FID is computed |8096|
|`num_vis_images`| Number of image visualized |1024|
|`batch_size`| Mini batch size |16|
|`num_workers`| Number of dataloader workers|8|
|`seed`| RNG seed |null|
|`save_epoch`| Epoch interval for checkpoint saves |1|
|`sanity_steps`| Validation sanity runs before training start |1|
|`max_epoch`| Maximum training epochs |250|
|`val_check_interval`| Epoch interval for evaluating metrics and saving generated samples |1|
|`resume`| Resume checkpoint |`null`|


References
==========
Official stylegan2-ada code and paper.

```
@article{Karras2019stylegan2,
    title   = {Analyzing and Improving the Image Quality of {StyleGAN}},
    author  = {Tero Karras and Samuli Laine and Miika Aittala and Janne Hellsten and Jaakko Lehtinen and Timo Aila},
    journal = {CoRR},
    volume  = {abs/1912.04958},
    year    = {2019},
}
```


License
=====================

Copyright © 2021 nihalsid

Permission is hereby granted, free of charge, to any person
obtaining a copy of this software and associated documentation
files (the “Software”), to deal in the Software without
restriction, including without limitation the rights to use,
copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the
Software is furnished to do so, subject to the following
conditions:

The above copyright notice and this permission notice shall be
included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
OTHER DEALINGS IN THE SOFTWARE.

