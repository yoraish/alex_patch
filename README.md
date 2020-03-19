# alex_patch - Alexnet for patch matching

## Setting up
* Clone the repository to your local machine.
* In the `/alex_net` directory, create the directories `build` and `models`.
* Download the model trace file [from this link](https://drive.google.com/file/d/1ooNeAaxV810csClIRIWIVAekmwHMTk3w/view?usp=sharing), and place it in the `models` folder. The name of that file should be `alexnet_fc_torchscript.pt`.
* Run the `make_alex_patch` bash script, with:
`./make_alex_patch`
* Execute inference on patches by running:
`./build/alex_patch`

#### Couple of notes
* Make sure that the absolute path to `libtorch` is specified in `make_alex_patch`.
