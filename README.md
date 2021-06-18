# _NIFA_ - Short-Term Cloud Coverage Prediction for Fruit Health Monitoring 

This work is sponsored by United States Department of Agriculture (USDA) with award number FAIN:20196702229922. 
It is part of a collaboration project titled "FACT: Deep Learning for Image-based Agriculture Evaluation" where Rutgers University is serving as the Prime. 
The goal of the project is to develop algorithms for assessing agriculture using visual imagery. 

Current module performs joint cloud segmentation and motion estimation from a sequence of sky images. 
The analysis use innovations in deep learning methods on large-scale datasets collected with ground-based sky imagers for future irradiance prediction. 
This work will be applied to cranberry crops at Philip E. Marucci Blueberry and Cranberry Research and Extension Center at RU.

The baseline network consists of two branches: a cloud motion estimation branch which is built on 
an unsupervised Siamese style recurrent spatial transformer network, and a cloud segmentation branch 
that is based on a fully convolutional network.

<!-- @import "[TOC]" {cmd="toc" depthFrom=1 depthTo=6 orderedList=false} -->

<!-- code_chunk_output -->

* [_NIFA_ - Short-Term Cloud Coverage Prediction for Fruit Health Monitoring](#README.md:1)
	* [Requirements](#requirements)
	* [Features](#features)
	* [Folder Structure](#folder-structure)
	* [Usage](#usage)
		* [Config file format](#config-file-format)
		* [Using config files](#using-config-files)
		* [Resuming from checkpoints](#resuming-from-checkpoints)
        * [Using Multiple GPU](#using-multiple-gpu)
        * [Tensorboard Visualization](#tensorboard-visualization)
	* [Contribution](#contribution)
	* [License](#license)
	* [Acknowledgements](#acknowledgements)

<!-- /code_chunk_output -->

## Requirements
* **Python >= 3.6**
* PyTorch >= 1.2
* OpenCV == 3.4.2
* Pillow >= 7.0.0
* tqdm (optional)
* tensorboard >= 1.14 (see [Tensorboard Visualization](#tensorboard-visualization))

To install the dependencies, go to the project root and run:
```shell
pip install -r requirements.txt --user
```

## Folder Structure

The folder structure and PyTorch boilerplate is adopted from 
[`pytorch-template`](https://github.com/victoresque/pytorch-template) by Victor Huang:

  ```
  FACT/
  │
  ├── train.py  - main script to start training
  ├── test.py   - evaluation of trained model
  │
  ├── parse_config.py - class to handle config file and cli options
  │
  ├── configs/
  │   ├── config_test.json  - configuration for testing with pretrained model
  │   ├── config_{...}.json - other configuration examples
  │
  ├── base/ - abstract base classes
  │   ├── base_data_loader.py
  │   ├── base_model.py
  │   └── base_trainer.py
  │
  ├── data_loader/ - anything about data loading goes here
  │   └── dataset_sccd.py   - dataset class for SCCD cloud data
  │   └── data_loaders.py   - data loaders 
  │   └── augmentation.py   - data augmentation operations
  │
  ├── model/ - models, losses, and metrics   
  │   ├── metric.py             - evaluation metrics
  │   └── loss.py               - training losses
  │   ├── model.py              - list of available models
  │   └── segmotionnet.py       - SegMotionNet model
  │   └── pwc_lite.py           - PWC-lite model
  │   └── correlation_package/  - custom correlation layer for PWC-lite
  │
  ├── trainer/ - trainers
  │   └── trainer.py    - process in charge of training and validation
  │   └── trainer_with_iterative_crf_refinement.py  - tentative process for weakly-supervised iterative learning
  │
  ├── logger/ - module for tensorboard visualization and logging
  │   ├── visualization.py
  │   ├── logger.py
  │   └── logger_config.json
  │  
  ├── utils/ - small utility functions
  │   ├── util.py
  │   └── ...
  │
  ├── saved/
  │   ├── models/ - trained models are saved here
  │   └── log/ - default logdir for tensorboard and logging output
  ```

## Usage

### Test
This repository provides a pre-trained model which can be downloaded from 
[here](https://drive.google.com/file/d/1Q8JVH6w3jpRwgpE-JafB_cQkuSRKsb9A/view?usp=sharing). 
Once downloaded, the file `segmotionnet_sccp-data_trained-with-backward-compat.pth` 
should be placed in the folder [`resources/saved_models`](resources/saved_models).

Additionally, this repository also contains some testing images in the folder [`resources/test_images`](resources/test_images).

To test the model, run:
  ```shell
  python test.py --config config/config_test.json --resume resources/saved_models/segmotionnet_sccp-data_trained-with-backward-compat.pth
  ```

Results will be saved in [`saved/`](saved) folder.

To simply infer the cloud coverage from an image, the following command is also provided:
  ```shell
  python estimate_cloud_coverage.py --image {path_to_image} --config config/config_test.json --resume resources/saved_models/segmotionnet_sccp-data_trained-with-backward-compat.pth
  ```
More details on how to use this script can be found in the PDF user manual in the [`doc/`](doc/) folder.

### Config file format
Config files are in `.json` format:
```javascript
{
  "name": "FACT_NIFAnet",                 // training session name
  "n_gpu": 1,                             // number of GPUs to use for training.
  
  "arch": {                               // architecture parameters  
    "type": "NifaNet",
    "args": {
      "in_ch": 3,                         // input image dimensions
      "out_ch": 2,
      "height": 112,
      "width": 112,
      "sampling_interval": 10.0,          // camera/pyranometer sampling interval (in seconds)
      "cloud_probability_threshold": 0.3, // threshold w.r.t. cloud probability
      "prediction_time_window": 300       // forecasting time window (in seconds)
    }              
  },
  "data_loader": {                        // dataset information
    "type": "SCCPDataLoader",
    "args": {
      "train_data_folders": [             // data folders
        "Rutgers_2020-08-22",
        "Rutgers_2020-09-01", ...
      ],
      "valid_data_folders": [
        ...
      ],
      "load_sun_positions": true,         // provided metadata
      "load_irradiances": true,
      "load_base_irradiances": true,
      "load_metadata": true,
      "batch_size": 2,                    // batch size
      "shuffle": true,                    // shuffle training data
      "num_workers": 2                    // number of cpu processes to be used for data loading
    }
  },
  "optimizer": {
    "type": "Adam",
    "args":{
      "lr": 0.001,                     // learning rate
      "weight_decay": 0,               // (optional) weight decay
      "amsgrad": true
    }
  },
  "loss": [                           // training losses
    {
      "type": "MRIFlowLoss",
      "args": {
        "factor_flow": 100.0,
        "factor_warp": 0.0
      }
    },
    {
      "type": "MRISegmentationLoss",
      "args": {
        "factor": 1.0
      }
    },
    {
      "type": "IrradiancePredLoss",
      "args": {
        "factor": 1.0,
        "criterion": "mape"
      }
    }
  ],
  "metrics": [
    "mape_irradiances",
    "frechet_distance_irradiances",
    "r2_score_irradiances"             // list of metrics to evaluate
  ],                         
  "lr_scheduler": {
    "type": "StepLR",                  // learning rate scheduler
    "args":{
      "step_size": 50,          
      "gamma": 0.1
    }
  },
  "trainer": {
    "epochs": 100,                     // number of training epochs
    "save_dir": "saved/",              // checkpoints are saved in save_dir/models/name
    "save_freq": 1,                    // save checkpoints every save_freq epochs
    "verbosity": 2,                    // 0: quiet, 1: per epoch, 2: full
  
    "monitor": "min val_loss"          // mode and metric for model performance monitoring. set 'off' to disable.
    "early_stop": 10	                 // number of epochs to wait before early stop. set 0 to disable.
  
    "tensorboard": true,               // enable tensorboard visualization
  }
}
```

### Using config files
Modify the configurations in `.json` config files, then run:

  ```
  python train.py --config config.json
  ```

### Resuming from checkpoints
You can resume from a previously saved checkpoint by:

  ```
  python train.py --resume path/to/checkpoint
  ```

### Using Multiple GPU
You can enable multi-GPU training by setting `n_gpu` argument of the config file to larger number.
If configured to use smaller number of gpu than available, first n devices will be used by default.
Specify indices of available GPUs by cuda environmental variable.
  ```
  python train.py --device 2,3 -c config.json
  ```
  This is equivalent to
  ```
  CUDA_VISIBLE_DEVICES=2,3 python train.py -c config.py
  ```

### Tensorboard Visualization
This template supports Tensorboard visualization by using either  `torch.utils.tensorboard` or [TensorboardX](https://github.com/lanpa/tensorboardX).

1. **Install**

    If you are using pytorch 1.1 or higher, install tensorboard by 'pip install tensorboard>=1.14.0'.

    Otherwise, you should install tensorboardx. Follow installation guide in [TensorboardX](https://github.com/lanpa/tensorboardX).

2. **Run training** 

    Make sure that `tensorboard` option in the config file is turned on.

    ```
     "tensorboard" : true
    ```

3. **Open Tensorboard server** 

    Type `tensorboard --logdir saved/log/` at the project root, then server will open at `http://localhost:6006`

By default, values of loss and metrics specified in config file, input images, and histogram of model parameters will be logged.

## Contribution
Feel free to contribute any kind of function or enhancement and to submit as pull requests.


## License
See [`LICENSE`](LICENSE) file.

## Acknowledgements
 - This project uses [`pytorch-template`](https://github.com/victoresque/pytorch-template) by Victor Huang as boilerplate.
 - The original *SegMotionNet* model was proposed and implemented by Chen Qin et al., c.f. 
 [`Joint-Learning-of-Motion-Estimation-and-Segmentation-for-Cardiac-MR-Image-Sequences`](https://github.com/cq615/Joint-Learning-of-Motion-Estimation-and-Segmentation-for-Cardiac-MR-Image-Sequences).
 - The original *PWC-lite* was implemented by Liang Liu et al., c.f. [`ARFlow`](https://github.com/lliuz/ARFlow).
 
