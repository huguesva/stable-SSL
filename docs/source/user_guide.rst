.. _user_guide:

.. currentmodule:: stable_ssl

.. automodule:: stable_ssl
   :no-members:
   :no-inherited-members:


User Guide
==========

This guide provides instructions for launching runs with ``stable-SSL``.

To make the process streamlined and efficient, we recommend using configuration files to define parameters and utilizing `Hydra <https://hydra.cc/>`_ to manage these configurations.

**General Idea.** ``stable-SSL`` is designed to minimize boilerplate code, providing a highly flexible framework with minimal hardcoded utilities. Most modules in the pipeline are modular and can instantiate objects from various sources, including ``stable-SSL``, ``PyTorch``, ``torchmetrics``, or even custom objects provided by the user. This allows you to seamlessly integrate your own components into the pipeline while leveraging the capabilities of ``stable-SSL``.


First step: define your base ``trainer`` module
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In ``stable-SSL``, the main ``trainer`` object must inherit from the ``BaseModel`` class. This class serves as the primary entry point for the training loop and provides all the essential methods required to train and evaluate your model effectively.

.. autosummary::
   :toctree: gen_modules/
   :template: myclass_template.rst

   BaseModel


More specifically, if you want to train a self-supervised learning model, you can inherit from the following classes, depending on the type of model you are interested in:


.. autosummary::
   :toctree: gen_modules/
   :template: myclass_template.rst

   JointEmbedding
   SelfDistillation

Here is what this instantiation looks like in the YAML configuration file:

.. code-block:: yaml

   _target_: stable_ssl.JointEmbedding


The objective to minimize (``objective``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``objective`` keyword is used to define the loss function for your model. ``stable-SSL`` offers a variety of loss functions to suit different training needs. Below is a list of the available loss functions:

.. autosummary::
   :toctree: gen_modules/
   :template: myclass_template.rst

   NTXEntLoss
   BYOLLoss
   VICRegLoss
   BarlowTwinsLoss

Here's an example of how to define the `objective` section in your YAML file:

.. code-block:: yaml

   objective:
      _target_: stable_ssl.NTXEntLoss
      temperature: 0.5


Optimization Configuration (``optim``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``optim`` keyword is used to define the optimization settings for your model. It allows users to specify both the ``optimizer`` object and the ``scheduler``. Below is an example configuration:


.. code-block:: yaml

   optim:
      epochs: 1000
      max_steps: 1000
      optimizer: 
         _target_: torch.optim.AdamW
         _partial_: True
         lr: 0.01
         weight_decay: 1e-6
      scheduler:
         _target_: torch.optim.lr_scheduler.OneCycleLR
         _partial_: True
         max_lr: 0.01
         epochs: ${trainer.optim.epochs}
         steps_per_epoch: ${eval:'${trainer.data._num_samples} // ${trainer.data.${trainer.train_on}.batch_size}'}


Hardware (``hardware``)
~~~~~~~~~~~~~~~~~~~~~~~~

Use the `hardware` keyword to configure hardware-related settings such as device, world_size (number of GPUs) or CPUs per task. An example is available below:

.. code-block:: yaml

   hardware:
      seed: 0
      float16: true
      device: "cuda:0"
      world_size: 1
      cpus_per_task: 8


Data Configuration (``data``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``data`` keyword defines data loading, preprocessing and data augmentation settings. Here is an example of how to define the ``data`` section in your YAML file:

.. code-block:: yaml

   train_on: base
   data:
      _num_classes: 10
      _num_samples: 50000
      base:
      _target_: torch.utils.data.DataLoader
      batch_size: 256
      drop_last: True
      shuffle: True
      num_workers: ${trainer.hardware.cpus_per_task}
      dataset:
         _target_: torchvision.datasets.CIFAR10
         root: ~/data
         train: True
         transform:
            _target_: stable_ssl.data.MultiViewSampler
            transforms:
            - _target_: torchvision.transforms.v2.Compose
               transforms:
                  - _target_: torchvision.transforms.v2.RandomResizedCrop
                  size: 32
                  scale:
                     - 0.2
                     - 1.0
                  - _target_: torchvision.transforms.v2.RandomHorizontalFlip
                  p: 0.5
                  - _target_: torchvision.transforms.v2.ToImage
                  - _target_: torchvision.transforms.v2.ToDtype
                  dtype: 
                     _target_: stable_ssl.utils.str_to_dtype
                     _args_: [float32]
                  scale: True
            - ${trainer.data.base.dataset.transform.transforms.0}
      test_out:
      _target_: torch.utils.data.DataLoader
      batch_size: 256
      num_workers: ${trainer.hardware.cpus_per_task}
      dataset:
         _target_: torchvision.datasets.CIFAR10
         train: False
         root: ~/data
         transform:
            _target_: torchvision.transforms.v2.Compose
            transforms:
            - _target_: torchvision.transforms.v2.ToImage
            - _target_: torchvision.transforms.v2.ToDtype
               dtype: 
                  _target_: stable_ssl.utils.str_to_dtype
                  _args_: [float32]
               scale: True

.. note::

   The ``train_on`` parameter specifies the dataset to train on. The ``data`` section can contain multiple datasets, each with its own configuration, but only one is used for training. The others can be used for evaluation.


Log Configuration (``logger``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``logger`` keyword is used to configure the logging settings for your run. 

One important section is ``metrics``, which lets you define the evaluation metrics to track during training. Metrics can be specified for each dataset, as shown below:


.. code-block:: yaml

   logger:
      base_dir: "./"
      level: 20
      checkpoint_frequency: 1
      every_step: 1
      metrics:
         base:
            acc1:
            _target_: torchmetrics.classification.MulticlassAccuracy
            num_classes: ${trainer.data._num_classes}
            top_k: 1
            acc5:
            _target_: torchmetrics.classification.MulticlassAccuracy
            num_classes: ${trainer.data._num_classes}
            top_k: 5
         test_out:
            acc1:
            _target_: torchmetrics.classification.MulticlassAccuracy
            num_classes: ${trainer.data._num_classes}
            top_k: 1
            acc5:
            _target_: torchmetrics.classification.MulticlassAccuracy
            num_classes: ${trainer.data._num_classes}
            top_k: 5


Network Configuration (``network``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``network`` keyword is used to define the settings of all the neural networks used, including the architecture of the backbone, projectors etc. Below is an example:

.. code-block:: yaml

   network:
      backbone:
         _target_: stable_ssl.utils.load_backbone
         name: resnet18
         dataset: "CIFAR10"
      projector:
         _target_: torch.nn.Sequential
         _args_:
            - _target_: torch.nn.Linear
            in_features: 512
            out_features: 2048
            bias: False
            - _target_: torch.nn.BatchNorm1d
            num_features: ${trainer.network.projector._args_.0.out_features}
            - _target_: torch.nn.ReLU
            - _target_: torch.nn.Linear
            in_features: ${trainer.network.projector._args_.0.out_features}
            out_features: 128
            bias: False
      projector_classifier:
         _target_: torch.nn.Linear
         in_features: 128
         out_features: ${trainer.data._num_classes}
      backbone_classifier:
         _target_: torch.nn.Linear
         in_features: 512
         out_features: ${trainer.data._num_classes}

The various components defined above can be accessed through the dictionary ``self.network`` in your trainer class. This allows you to define the forward pass, compute losses, and specify evaluation metrics efficiently.
