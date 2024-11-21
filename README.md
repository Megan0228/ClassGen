# Bridging Domain and Modality Gaps: A Class Enhancement Approach for Unsupervised Adaptation in Vision-Language Models
 
This is the PyTorch code of the [ClassGen paper](). The repository supports finetuning a CLIP model on unlabeled images from a target domain.


### Dataset Setup
Dataset paths are stored in [dataset_catalog.json](https://github.com/salesforce/MUST/blob/main/dataset_catalog.json), which need to be modified to local paths. The imagenet dataset follows the standard folder structure. For other datasets, please refer to the scrips from [VISSL](https://github.com/facebookresearch/vissl/tree/main/extra_scripts/datasets) to download and prepare. CLIP's labels and prompt templates are stored in [classes.json](https://github.com/salesforce/MUST/blob/main/classes.json) and [templates.json](https://github.com/salesforce/MUST/blob/main/templates.json).





