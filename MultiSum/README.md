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
