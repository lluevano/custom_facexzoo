# conventional_training

## Particularities

Many example configurations are available in the ".job" scripts. To submit these scripts:
1. Run "SETSHELL grid".
2. Modify your personal grid parameters and training parameters in your ".job" script
3. Run the qsub command as `qsub -l sgpu example_script.job` for using a SGE Short GPU.

### Output directory convention
The output directory convention is important when using the standalone evaluation scripts at the "../../test_protocol/" folder.
The convention is: `"{loss_function}_nf{n_unfrozen_layers}_lr_{lr}"`

### Fine tuning
To fine-tune to a different dataset, set the "--fine_tune" or "-ft" parameter to True.
This allows to discard the head weight (used in the loss function) to successfully load a previous checkpoint.

### Previous modules
The "train.py" file is able to load pytorch module structures that can be trained and used for evaluations. This is controlled by the "--module_type" parameter. See "../../modules" for module options.

### Freezing and unfreezing layers
This function is controlled by the "--n_unfrozen_layers" parameter and only available with the "fine tuning" option. If set to 0, all the layers in the network will be trained. If set to a different number, it will freeze the first N layers of the network. If a previous module is defined, this will also freeze the first layers counting the ones from the previous module to the backbone and the head weight, and then unfreeze all the previous module layers.

### Bob evaluations at training time
Bob evaluations can be performed by specifying a Bob dataset as defined in the "../../test_protocol/bob_test_protocol.py" for dev and eval sets. Currently only the options "survface" and "tinyface" are available.

* * *
The rest of this README file comes by default from the original FaceXZoo repository.

## train.py
The common training pipeline.
## train_amp.py
The mixed precision training pipeline. You should install the [apex](https://github.com/nvidia/apex) firstly.