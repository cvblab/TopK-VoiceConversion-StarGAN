# Top-K Many-to-Many Voice Conversion with StarGAN: Choosing Only the Best Voice Imitators
Repository containing files for training deep learning models for the introduction of the Top-K methodology for the training of the training of Voice Conversion models such as StarGAN-VC and StarGAN-VCv2.

## Installation

* Install in your enviroment a compatible torch version with your GPU. For example:
```
git clone https://github.com/cvblab/TopK-VoiceConversion-StarGAN.git
pip install -r requirements.txt
```

## Datasets

For proper usage, training and validation of the proposed Top-K method, please download and pre-process the following dataset as indicated:

* [VCTK](https://homepages.inf.ed.ac.uk/jyamagis/page3/page58/page58.html)

Example with VCTK:

```shell script
mkdir ../data/VCTK-Data
wget https://datashare.is.ed.ac.uk/bitstream/handle/10283/2651/VCTK-Corpus.zip?sequence=2&isAllowed=y
unzip VCTK-Corpus.zip -d ../data/VCTK-Data
```

If the downloaded VCTK is in tar.gz, run this:

```shell script
tar -xzvf VCTK-Corpus.tar.gz -C ../data/VCTK-Data
```

## Preprocessing data

We will use Mel-Cepstral coefficients(MCEPs) here.

Example script for VCTK data which we can resample to 22.05kHz. The VCTK dataset is not split into train and test wavs, so we perform a data split.

```shell script
# VCTK-Data
python preprocess.py --perform_data_split y \
                     --resample_rate 16000 \
                     --origin_wavpath ../data/VCTK-Data/VCTK-Corpus/wav48 \
                     --target_wavpath ../data/VCTK-Data/VCTK-Corpus/wav16 \
                     --mc_dir_train ../data/VCTK-Data/mc/train \
                     --mc_dir_test ../data/VCTK-Data/mc/test \
                     --speakers p262 p272 p229 p232
```

# Training

Example script:

```shell script
# example with StarGAM-VC (v1), Top-K training from iter 25000 topk_gamma 0.9999 topk_v 0.2

python main.py --stargan_version v1 \
                --topk_training True \
                --preload_data True \
                --mlflow_experiment_name TopKv1_FINALEXP \
                --where_exec local \
                --topk_from_iter 25000 \
                --topk_gamma 0.9999 \
                --topk_v 0.2
```
## Conversion

Example script:

```shell script

python convert_for_evaluation.py \
        --num_converted_wavs 8 \
        --models_save_dir PATH_WHERE_MODELS_ARE_STORED \
        --convert_dir PATH_WHERE_CONVERSIONS_WILL_BE_STORED
        
```

## Evaluation

Example script:

```shell script
# Extract MCD

python evaluation_converted_mcds.py \
        --converted_samples_data_dir PATH_WHERE_CONVERSIONS_ARE_STORED
        
# Extract MOS

python evaluation_converted_mosnet.py \
        --converted_samples_data_dir PATH_WHERE_CONVERSIONS_ARE_STORED
        
```

## Tables and plots

Example script:

```shell script
# Extract MCD Plots
python extract_tables_plots_evaluation_results.py \
        --converted_samples_data_dir PATH_WHERE_CONVERSIONS_ARE_STORED
        
# Extract MOSNet Plots
python extract_tables_plots_evaluation_results_mosnet.py \
        --converted_samples_data_dir PATH_WHERE_CONVERSIONS_ARE_STORED
                
        
```


