# MMSum: A Dataset for Multimodal Summarization and Thumbnail Generation of Videos

In CVPR 2024 [paper](https://arxiv.org/abs/2306.04216)

Project page: [link](https://mmsum-dataset.github.io/)

## Usage

### Set up Environment

Create a conda virtual environment and activate it.

```
conda env create -f MMSum/mmsum.yml
conda activate mmsum
```

### Download Datasets and Preprocessing

Please download the video, keyframe, and thumbnail features from this [link](https://drive.google.com/drive/folders/1ZE3p7JmoZe0EK-HIxpKrYUdHqXwFabUf?usp=sharing).

We provide the scripts [keyframe_feature.py, text_feature.py, video_feature.py] to create the features for the whole text environment.
We provide code to generate the segmented text environment data in seg_video_feature.py and video_feature_mmsum.py.

After preprocessing, there should be 2 sets of training, validation, test sets.
{set_name}.tsv is for the whole sentence and video environment and {set_name}_2.tsv is for the segmented video environment.

Please place them in MMSum/src/data.

If you want to know how the data was collected, please go to our [data collection repo](https://github.com/Jason-Qiu/MMSum_Data_Collection_Tool).

### Training and Evaluation

For training, please cd into MMSum/src/runtime and run the following command:

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

### Citation

```
@inproceedings{Qiu2023MMSumAD,
    title={MMSum: A Dataset for Multimodal Summarization and Thumbnail Generation of Videos},
    author={Jielin Qiu and Jiacheng Zhu and William Han and Aditesh Kumar and Karthik Mittal
            and Claire Jin and Zhengyuan Yang and Linjie Li and Jianfeng Wang
            and Ding Zhao and Bo Li and Lijuan Wang},
    journal={CVPR},
    year={2024}
```

### Data format

In each JSON file in `annotation.tar.gz` from the above [link](https://drive.google.com/drive/folders/1ZE3p7JmoZe0EK-HIxpKrYUdHqXwFabUf?usp=sharing), we include the metadata info for each video. For example, below is the info from ANIAMP0000.json. The `info` section contains `video_id`, `youtube_id`, `url`, `author`, `title`, `num_of_segments`, `duration`, `category`, `sub_category`. Due to Intellectual property, we didn't include the raw video in the [link](https://drive.google.com/drive/folders/1ZE3p7JmoZe0EK-HIxpKrYUdHqXwFabUf?usp=sharing), but you can use `url` to download the original video from Youtube. 

```
{
  "info": {
    "video_id": "ANIAMP0000",
    "youtube_id": "XI8GPsf6TAc",
    "url": "https://youtube.com/watch?v=XI8GPsf6TAc",
    "author": "Happy Learning English",
    "title": "Amphibians | Educational Video for Kids",
    "num_of_segments": 3,
    "duration": "00:04:30",
    "category": "animals",
    "sub_category": "amphibians"
  },
  "summary": [
    {
      "segment": 0,
      "start_time": "00:00:24",
      "summary": "THE ANPHIBIANS",
      "end_time": "00:01:25",
      "length": "00:01:01"
    },
    {
      "segment": 1,
      "start_time": "00:01:26",
      "summary": "OVIPAROUS",
      "end_time": "00:02:47",
      "length": "00:01:21"
    },
    {
      "segment": 2,
      "start_time": "00:02:48",
      "summary": "CARNIVORES",
      "end_time": "03:00:00",
      "length": "00:01:42"
    }
  ],
  "transcript": [
    {
      "index": 0,
      "start_time": "00:00:06",
      "end_time": "00:00:12",
      "length": "00:00:06",
      "summary": "Hello everybody! Today we\u2019re going to look\nat a truly amazing group of vertebrates..."
    },
    {
      "index": 1,
      "start_time": "00:00:13",
      "end_time": "00:00:18",
      "length": "00:00:05",
      "summary": "When they\u2019re born they usually live in water...\nbut when they grow up and become adults they"
    },
    {
      "index": 2,
      "start_time": "00:00:19",
      "end_time": "00:00:24",
      "length": "00:00:05",
      "summary": "spend most of their time on land. We present\n- the Amphibians!"
    },
    {
      "index": 3,
      "start_time": "00:00:58",
      "end_time": "00:01:03",
      "length": "00:00:05",
      "summary": "All amphibians have some common characteristics\nthat you should know about so you can recognize"
    },
    {
      "index": 4,
      "start_time": "00:01:04",
      "end_time": "00:01:10",
      "length": "00:00:06",
      "summary": "and differentiate them.\nAmphibians have thin, bare skin, with no hairs"
    },
    {
      "index": 5,
      "start_time": "00:01:11",
      "end_time": "00:01:17",
      "length": "00:00:06",
      "summary": "and scales to protect them. Most have four\nlegs and a membrane between their toes that"
    },
    {
      "index": 6,
      "start_time": "00:01:18",
      "end_time": "00:01:20",
      "length": "00:00:02",
      "summary": "allows them to move much better in the water."
    },
    {
      "index": 7,
      "start_time": "00:01:25",
      "end_time": "00:01:27",
      "length": "00:00:02",
      "summary": "Amphibians are oviparous, but they don\u2019t"
    },
    {
      "index": 8,
      "start_time": "00:01:28",
      "end_time": "00:01:33",
      "length": "00:00:05",
      "summary": "incubate their eggs after laying them... they\nabandon them and don\u2019t care for their young."
    },
    {
      "index": 9,
      "start_time": "00:01:34",
      "end_time": "00:01:38",
      "length": "00:00:04",
      "summary": "Not very good parents, huh?\nWhen they hatch, they\u2019re small larvae and"
    },
    {
      "index": 10,
      "start_time": "00:01:39",
      "end_time": "00:01:46",
      "length": "00:00:07",
      "summary": "live in water. Slowly... very slowly... their\nbodies go through a process called metamorphosis."
    },
    {
      "index": 11,
      "start_time": "00:01:48",
      "end_time": "00:01:54",
      "length": "00:00:06",
      "summary": "During this process, the body of the amphibian...\nchanges... their front and rear legs, their"
    },
    {
      "index": 12,
      "start_time": "00:01:55",
      "end_time": "00:02:02",
      "length": "00:00:07",
      "summary": "limbs, grow... and their heads and their bodies\ndevelop, so they finally look like their parents."
    },
    {
      "index": 13,
      "start_time": "00:02:05",
      "end_time": "00:02:11",
      "length": "00:00:06",
      "summary": "In the early stages of their lives... amphibians\nbreathe through gills, but when they grow"
    },
    {
      "index": 14,
      "start_time": "00:02:12",
      "end_time": "00:02:14",
      "length": "00:00:02",
      "summary": "up and become adults... they breathe with\ntheir lungs. "
    },
    {
      "index": 15,
      "start_time": "00:02:16",
      "end_time": "00:02:22",
      "length": "00:00:06",
      "summary": "The problem is, their lungs are very small, and cannot get all the oxygen they need to live. "
    },
    {
      "index": 16,
      "start_time": "00:02:23",
      "end_time": "00:02:24",
      "length": "00:00:01",
      "summary": "But nature is very clever..."
    },
    {
      "index": 17,
      "start_time": "00:02:25",
      "end_time": "00:02:29",
      "length": "00:00:04",
      "summary": "and has solved this problem by allowing them\nto breathe and get the oxygen they need..."
    },
    {
      "index": 18,
      "start_time": "00:02:30",
      "end_time": "00:02:37",
      "length": "00:00:07",
      "summary": "through their skin. That\u2019s why they need\nto be near water - to keep their skin wet."
    },
    {
      "index": 19,
      "start_time": "00:02:40",
      "end_time": "00:02:43",
      "length": "00:00:03",
      "summary": "In the early stages of their life, some amphibians are herbivores, "
    },
    {
      "index": 20,
      "start_time": "00:02:44",
      "end_time": "00:02:48",
      "length": "00:00:04",
      "summary": "but when they grow up... most become carnivores."
    },
    {
      "index": 21,
      "start_time": "00:02:52",
      "end_time": "00:02:55",
      "length": "00:00:03",
      "summary": " So they need to hunt..."
    },
    {
      "index": 22,
      "start_time": "00:03:02",
      "end_time": "00:03:04",
      "length": "00:00:02",
      "summary": "Some have a long, sticky tongue they shoot"
    },
    {
      "index": 23,
      "start_time": "00:03:05",
      "end_time": "00:03:07",
      "length": "00:00:02",
      "summary": "out to capture prey."
    },
    {
      "index": 24,
      "start_time": "00:03:08",
      "end_time": "00:03:10",
      "length": "00:00:02",
      "summary": "Aren\u2019t amphibians fascinating?"
    },
    {
      "index": 25,
      "start_time": "00:03:11",
      "end_time": "00:03:13",
      "length": "00:00:02",
      "summary": " And also a bit strange"
    },
    {
      "index": 26,
      "start_time": "00:03:18",
      "end_time": "00:03:22",
      "length": "00:00:04",
      "summary": "So let\u2019s remember the most important characteristics..."
    },
    {
      "index": 27,
      "start_time": "00:03:23",
      "end_time": "00:03:30",
      "length": "00:00:07",
      "summary": "Amphibians are vertebrates; they\u2019re oviparous;\nin the early stages of their life they live"
    },
    {
      "index": 28,
      "start_time": "00:03:31",
      "end_time": "00:03:32",
      "length": "00:00:01",
      "summary": "in water as larvae, "
    },
    {
      "index": 29,
      "start_time": "00:03:36",
      "end_time": "00:03:39",
      "length": "00:00:03",
      "summary": "but slowly they change\nuntil they look just like their parents. "
    },
    {
      "index": 30,
      "start_time": "00:03:40",
      "end_time": "00:03:43",
      "length": "00:00:03",
      "summary": "This process of change is called metamorphosis."
    },
    {
      "index": 31,
      "start_time": "00:03:47",
      "end_time": "00:03:49",
      "length": "00:00:02",
      "summary": "Amphibians are carnivores, so they have to"
    },
    {
      "index": 32,
      "start_time": "00:03:50",
      "end_time": "00:03:55",
      "length": "00:00:05",
      "summary": "hunt to eat; they have thin, smooth skin,\nand breathe through their skin and with their"
    },
    {
      "index": 33,
      "start_time": "00:03:56",
      "end_time": "00:03:59",
      "length": "00:00:03",
      "summary": "lungs.\nAmphibians are so interesting, aren\u2019t they?"
    },
    {
      "index": 34,
      "start_time": "00:04:00",
      "end_time": "00:04:05",
      "length": "00:00:05",
      "summary": "Goodbye for now everyone, and don\u2019t forget\nto subscribe to Happy Learning!"
    }
  ]
}
```

### Acknowledgements

We appreciate the public codebase from [MLASK](https://github.com/ufal/MLASK).

## License

This project is licensed under the CC BY-NC-SA License.

