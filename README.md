# For reviewers:

You
can [reproduce graphs and tables](#reproduce-graphs-and-tables), [download images and results](#download-datasets-and-images), [visualize attacked datasets](#visualize-attacked-datasets),
and [replicate experiments](#replicate-experiments): the randomized attack in the paper is named `random` and the
deliberate attacks selecting easier targets and perturbing arbitrary regions are named `biased` and `arbitrary`
respectively.

Home directory is `$USER_DIR` and the working directory
is `$PROJECT_DIR/obfuscation`, i.e. all relative paths begin there unless there is an explicit `cd` command. Replace
these stand-in names with your own directories to run the commands.

## Reproduce graphs and tables

The code store the general trends as `*.csv` and individual results as `*.parquet`
in `./data/randomized/results`, `./data/biased/results` and `./data/arbitrary/results`.

Use knitr on `./analysis/randomized.Rmd`, `./analysis/biased.Rmd`  and `./analysis/arbitrary.Rmd` to produce
graphs in `./analysis/imgs/` and tables in `./analysis/randomized.tex`, `./analysis/biased.tex`,
and `./analysis/arbitrary.tex`. You get either generate a latex or pdf document and change the `norm` parameter
to `0.05` to analyze experiments with max norm.

Knitting `./analysis/summary.Rmd` to `./analysis/summary.tex` produce the hypothesis
and model summaries.

I use R 4.4.0 together with renv to install the required packages. You can use `renv::restore()` to
[reproduce](https://rstudio.github.io/renv/articles/renv.html) the R environment.

## Download images and results

Sample datasets and attacked images are included with the code in `./sample`. To browse and download all datasets and
images, please go to the [google cloud storage bucket](https://console.cloud.google.com/storage/browser/intent-obfusc) (
it's public though may still need to sign in). You can browse the bucket online or even download the ~80GB bucket
using [these instructions online](https://cloud.google.com/storage/docs/access-public-data).

The attacked images are stored in `./data/randomized/images`, `./data/arbitrary/images`,
and `./data/biased/images` and the datasets are saved in `./data/randomized/datasets`, `./data/arbitrary/datasets`,
and `./data/biased/datasets`.

## Visualize attacked datasets

Minimal code to visualize a dataset (on the original images):

1. Install packages

   ```bash
   pip install fiftyone
   ```

2. Download COCO

   ```python
   import fiftyone.zoo as foz
   dataset = foz.load_zoo_dataset("coco-2017", split="validation")  # only validation required and downloaded to ~/fiftyone/coco-2017/validation
   ```

3. Import dataset

   ```python
   # https://docs.voxel51.com/user_guide/dataset_creation/datasets.html#fiftyonedataset
   import fiftyone as fo

   dataset_dir = "obfuscation/sample/randomized/datasets/itr_200_norm_None_repeat_20_mislabel_bbox_ssd_512_coco"
   
   dataset = fo.Dataset.from_dir(
       dataset_dir=dataset_dir,
       dataset_type=fo.types.FiftyOneDataset,
       rel_dir="~/fiftyone/coco-2017/validation/data",
   )
   
   session = fo.launch_app(dataset)
   ```   

   Then point browser to `localhost:5151`. Select the dataset in the top bar. The attack and success images are in
   the `sample tags` sidebar, the perturb and target objects are in the `label tags` sidebar, and prediction
   results (`pgd`) and arbitrary perturb bbox (in the arbitrary experiment only) are in the `labels` sidebar.

# Replicate experiments

Adapt as needed. I use miniconda to install conda packages and bash to run shell commands on the internal HPC. The exact
commands to reproduce the experiments (including the random seeds) are included in `./slurm`

## Declare variables

`USER_DIR` is the conda location and `PROJECT_DIR` is the project directory.

```bash
USER_DIR=[replace]
PROJECT_DIR=[replace]
```

## Patch mmdetection

Already patched in `./mmdetection`:
Download [mmdetection 2.28.2](https://github.com/open-mmlab/mmdetection/releases/tag/v2.28.2) and rename
to `mmdetection`. Patch `mmdetection` to retrieve class probabilities and remove data augmentation by running

```bash
cd $PROJECT_DIR/changelist
chmod +x move.sh
./move.sh
```

## Download COCO

Download COCO 2017 validation images using

```bash
cd $PROJECT_DIR

mkdir coco 
cd coco

wget http://images.cocodataset.org/zips/val2017.zip
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip

unzip val2017.zip
unzip annotations_trainval2017.zip

rm val2017.zip
rm annotations_trainval2017.zip
```

## Install packages

Install conda and create new environment:

```bash
mkdir -p $USER_DIR/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-py38_23.11.0-2-Linux-x86_64.sh -O $USER_DIR/miniconda3/miniconda.sh
bash $USER_DIR/miniconda3/miniconda.sh -b -u -p $USER_DIR/miniconda3
rm -rf $USER_DIR/miniconda3/miniconda.sh

$USER_DIR/miniconda3/bin/conda init

conda create --name coco python=3.8 -y
conda activate coco
```

Install packages

```bash
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 pyarrow tqdm pytest ipython -c pytorch -c nvidia -c conda-forge -y
```

Install fiftyone

```bash
pip install fiftyone==0.23.2
```

Install mmcv

```bash
pip install openmim
mim install mmcv-full==1.7.2
```

Install mmdet in developer mode

```bash
cd $PROJECT_DIR/mmdetection
pip install -v -e .
```

Download model checkpoints

```bash
mkdir -p checkpoints && cd checkpoints
mim download mmdet --config faster_rcnn_r50_fpn_1x_coco --dest .
mim download mmdet --config cascade_rcnn_r50_fpn_1x_coco --dest .
mim download mmdet --config retinanet_r50_fpn_1x_coco --dest .
mim download mmdet --config ssd512_coco --dest .
mim download mmdet --config yolov3_d53_mstrain-608_273e_coco --dest .
```

Check mmdet

```bash
python ../demo/image_demo.py ../demo/demo.jpg faster_rcnn_r50_fpn_1x_coco.py faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth --device cuda:0 --out-file result.jpg
```

## Setup mongodb

Install mongodb to run fiftyone

```bash
conda create --name db -y 
conda activate db
conda install -c conda-forge mongodb==6.0.12 -y
```

Check mongodb:

1. launch mongod server

   ```bash
   conda activate db
   mkdir -p ~/db
   mongod --dbpath ~/db --port 12345 
   ```

2. connect with fiftyone

   ```bash
   conda activate coco
   export FIFTYONE_DATABASE_URI=mongodb://127.0.0.1:12345/
   ```

   launch python

   ```python
   import fiftyone as fo
   ```

## Only the code

Adapt as needed:

```bash
USER_DIR=[replace]
PROJECT_DIR=[replace]

cd $PROJECT_DIR

mkdir coco 
cd coco

wget http://images.cocodataset.org/zips/val2017.zip
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip

unzip val2017.zip
unzip annotations_trainval2017.zip

rm val2017.zip
rm annotations_trainval2017.zip

mkdir -p $USER_DIR/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-py38_23.11.0-2-Linux-x86_64.sh -O $USER_DIR/miniconda3/miniconda.sh
bash $USER_DIR/miniconda3/miniconda.sh -b -u -p $USER_DIR/miniconda3
rm -rf $USER_DIR/miniconda3/miniconda.sh

$USER_DIR/miniconda3/bin/conda init

conda create --name coco python=3.8 -y
conda activate coco

conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 pyarrow tqdm pytest ipython -c pytorch -c nvidia -c conda-forge -y

pip install fiftyone==0.23.2

pip install openmim
mim install mmcv-full==1.7.2

cd $PROJECT_DIR/mmdetection
pip install -v -e .

mkdir -p checkpoints && cd checkpoints
mim download mmdet --config faster_rcnn_r50_fpn_1x_coco --dest .
mim download mmdet --config cascade_rcnn_r50_fpn_1x_coco --dest .
mim download mmdet --config retinanet_r50_fpn_1x_coco --dest .
mim download mmdet --config ssd512_coco --dest .
mim download mmdet --config yolov3_d53_mstrain-608_273e_coco --dest .

python ../demo/image_demo.py ../demo/demo.jpg faster_rcnn_r50_fpn_1x_coco.py faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth --device cuda:0 --out-file result.jpg

conda create --name db -y 
conda activate db
conda install -c conda-forge mongodb==6.0.12 -y

```

## Run scripts

My internal HPC uses [slurm](https://slurm.schedmd.com/documentation.html). The entry point to the code
is `./src/run.py`. The defaults are given in `./configs/runtime/vanish_bbox.py`. The scripts below simply batches the
experiment.

1. (Already generated.) Generate attack configs to `./configs` and slurm scripts to `./slurm`:

   ```bash
   cd ./configs && chmod +x generate_config.sh 
   ./generate_config.sh faster_rcnn yolo_v3 retinanet ssd_512 cascade_rcnn  

   cd ./slurm && chmod +x gen_scripts.sh
   
   # random
   chmod +x randomized.sh && ./randomized.sh
      
   # arbitrary
   chmod +x arbitrary.sh && ./arbitrary.sh
   
   # biased sampler
   chmod +x biased.sh && ./biased.sh
   ```

2. Run scripts using sbatch (change mongod `DB_ENV` and pytorch `TORCH_ENV` environment names in `./run_batch.sh`):

   ```bash
   cd $PROJECT_DIR/obfuscation/slurm && mkdir -p logs && sbatch ./randomized/run_batch.sh
   cd $PROJECT_DIR/obfuscation/slurm && mkdir -p logs && sbatch ./arbitrary/run_batch.sh
   cd $PROJECT_DIR/obfuscation/slurm && mkdir -p logs && sbatch ./biased/run_batch.sh
   ```

   Analysis results, datasets, images and logs will be saved in the corresponding directories passed to
   the `./gen_scripts.sh` command.

# Diagnostics

## Run pytests

[Launch mongodb server](#setup-mongodb) and then run

```bash
conda activate coco
export FIFTYONE_DATABASE_URI=mongodb://127.0.0.1:12345/ 
cd $PROJECT_DIR/obfuscation/
python -m pytest ./test
```

## Check slurm logs

```bash
cd $PROJECT_DIR/obfuscation/slurm/logs
```

# Credits

The code includes a patched [mmdetection 2.28.2](https://github.com/open-mmlab/mmdetection/releases/tag/v2.28.2)
directory, which contains an APACHE license.