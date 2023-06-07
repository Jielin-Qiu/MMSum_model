# MultiSum: A Dataset for Multimodal Summarization and Thumbnail Generation of Videos

## Usage

### Set up Environment

Create a conda virtual environment and activate it.

```
conda env create -f MultiSum/multisum.yml
conda activate multisum
```

### Download Datasets

Please download the preprocessed data from this [link](https://drive.google.com/drive/folders/1QRtU32oiTbAzU9swn1ZFFg0hxmpz3a3k?usp=sharing).

There are 2 sets of training, validation, test sets.
{set_name}.tsv is for the whole sentence and video environment and {set_name}_2.tsv is for the segmented video environment.

Due to file length limits, we currently only provide the whole sentence environment.

Please place them in MultiSum/src/data.

There are also 2 sets of folders with features from keyframes, videos, and thumbnails.
Similarly to the tsv files, {feature_name} is for the whole sentence and video environment and {feature_name}2 is for the segmented environment.

Due to file length limits, we currently only provide the whole sentence environment.

Please place them in MultiSum/src/data.

We provide code to generate the segmented environment data in seg_video_feature.py and video_feature_multisum.py.



### Preprocessing data

All preprocessing code is in the preprocessing folder.

### Training and Evaluation

For training, please cd into MultiSum/src/runtime and run the following command:

```
sh train.sh
```

For evaluation, stay in the same folder and run the following command:

```
sh eval.sh
```

For thumbnail generation, stay in the same folder and run the following command:

```
python generate_thumbnail.py
```


### Acknowledgements

We appreciate the public codebase from [MLASK](https://github.com/ufal/MLASK).

## License

This project is licensed under the CC BY-NC-SA License.
