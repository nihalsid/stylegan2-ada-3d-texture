# Texturify Generating Textures on 3D Shape Surfaces

## Dependencies

Install python requirements:

```commandline
pip install -r requirements.txt
pip install torch==1.10.2+cu113 torchvision==0.11.3+cu113 torchaudio==0.10.2+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
```

Install trimesh from our fork:
```bash
cd ~
git clone git@github.com:nihalsid/trimesh.git
cd trimesh
python setup.py install
```

Also, for differentiable rendering we use `nvdiffrast`. You'll need to install its dependencies:

```bash
sudo apt-get update && sudo apt-get install -y --no-install-recommends \
    pkg-config \
    libglvnd0 \
    libgl1 \
    libglx0 \
    libegl1 \
    libgles2 \
    libglvnd-dev \
    libgl1-mesa-dev \
    libegl1-mesa-dev \
    libgles2-mesa-dev \
    cmake \
    curl
```

Install `nvdiffrast` from official source:

```bash
cd ~ 
git clone git@github.com:NVlabs/nvdiffrast.git
cd nvdiffrast
pip install .
```

Apart from this, you will need approporiate versions of torch-scatter, torch-sparse, torch-spline-conv, torch-geometric, depending on your torch+cuda combination. E.g. for torch-1.10 + cuda11.3 you'd need:  

```commandline
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.10.0+cu113.html
```
## Dataset

From project root execute:
```bash
mkdir data
cd data
wget https://www.dropbox.com/s/or9tfmunvndibv0/data.zip
unzip data.zip
```

For custom data processing check out https://github.com/nihalsid/CADTextures

## Output Directories

Create a symlink `runs` in project root from a directory `OUTPUTDIR` where outputs would be stored  
```bash
ln -s OUTPUTDIR runs
```

## Running Experiments

Configuration provided with hydra config file `config/stylegan2.yaml`. Example training:

```bash
python trainer/train_stylegan_real_feature.py wandb_main=False val_check_interval=5 experiment=test_run lr_d=0.001 sanity_steps=1 lambda_gp=14 image_size=512 batch_size=4 num_mapping_layers=5 views_per_sample=2 g_channel_base=32768 random_bg=grayscale num_vis_images=256 preload=False dataset_path=data/Photoshape/shapenet-chairs-manifold-highres-part_processed_color mesh_path=data/Photoshape/shapenet-chairs-manifold-highres pairmeta_path=data/Photoshape-model/metadata/pairs.json image_path=data/Photoshape/exemplars mask_path=data/Photoshape/exemplars_mask
```

## Checkpoints

Available [here](https://www.dropbox.com/scl/fi/cz9arygdbz05gucapldd1/texturify_checkpoints.zip?rlkey=n19t8x0zq13i7hmodfnjkgrst&dl=0).

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

