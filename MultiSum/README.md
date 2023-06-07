# MLASK <a href="https://en.wiktionary.org/wiki/mlaska%C4%87"><img src="https://fonts.gstatic.com/s/e/notoemoji/latest/1f60b/512.gif" alt="😋" width="32" height="32"></a>


Code ~~and data~~ for the EACL 2023 (findings) paper: "[MLASK: Multimodal Summarization of Video-based News Articles](https://aclanthology.org/2023.findings-eacl.67/)".

![MLASK-overview](./resources/model_overview.png?raw=true)


## Data

The MLASK corpus consists of 41,243 multi-modal documents – video-based news articles in the Czech language – collected from [Novinky.cz](https://www.novinky.cz/) and [Seznam Zprávy](https://www.seznamzpravy.cz/). 

Each document consists of:

- a .mp4 video (up to 5 minutes)
- a single image (cover picture)
- the article's text
- the article's summary
- the article's title
- the article's publication date

***Stay tuned - the dataset release is coming soon!***

## Code

We include the code used in our experiments. It is structured as follows:

```
├── feature_extraction
│   ├── extract_image_features.ipynb - Image feature extraction (Section 4.2)
│   └── extract_video_features.ipynb - Video feature extraction (Section 4.2)
└── src
    ├── model
    │   ├── mms_modeling_t5.py - Modified version of the mT5 model, that includes video encoder, image encoder etc (Section 4)
    │   └── model_mms.py - Implementation of training loop, evaluation metrics and logging
    ├── data
    │   ├── data_laoder.py - Implementation of data loader/data pre-processing
    │   └── utils.py - Utility functions
    └── runtime
        ├── test_mms_model.py - MMS model evaluation (Section 5.2 and 5.3)
        └── train_mms_model.py - MMS model training (Section 5.2 and 5.3)

```

`RougeRaw.py` required by `model_mms.py` can be downloaded from the [SumeCzech repository](https://lindat.cz/repository/xmlui/handle/11234/1-2615?locale-attribute=cs).

## License

Our code is released under Apache License 2.0, unless stated otherwise.

## Citation

If you find our code ~~or data~~ useful, please cite:
```
@inproceedings{krubinski-pecina-2023-mlask,
    title = "{MLASK}: Multimodal Summarization of Video-based News Articles",
    author = "Krubi{\'n}ski, Mateusz  and Pecina, Pavel",
    booktitle = "Findings of the Association for Computational Linguistics: EACL 2023",
    month = may,
    year = "2023",
    address = "Dubrovnik, Croatia",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.findings-eacl.67",
    pages = "880--894",
}
```
