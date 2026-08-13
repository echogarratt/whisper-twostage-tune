# Whisper two-step tune

Based on [1], this code provides a skeleton on which to perform fine tuning experiments with Whisper models, using an intermediate step. There are also setup files to create intermediate datasets using the CSM-1B [citation needed] Neural Audio Codec.

## Description

An in-depth paragraph about your project and overview of use.

### Two step tuning

The backbone of two step tuning is `training_pipeline.py`. This provides some relatively sensible default values for tuning, and some default locations for datasets. These are hardcoded and you must update them if using this script.

There are two entrypoints to the fine tuning, either from the terminal with command line arguments, or using an entrypoint python file. `training_example.py` is a minimal entrypoint example with explanations of how to run tuning by both command line and from python. A more fully-featured entrypoint is `run.py` which takes a configuration ID flag to specify the configuration and speaker to be tuned. This is useful when doing a Job Array in the SLURM Scheduler on a HPC system, where all of the configurations can take their own job and so make best use of HPC resources.




## Getting Started

### Dependencies

* Describe any prerequisites, libraries, OS version, etc., needed before installing program.
* ex. Windows 10

### Executing program

* How to run the program
* Step-by-step bullets
```
code blocks for commands
```

## Authors

Echo Garratt\
School of Computer Science\
University of Sheffield\
edgarratt1@sheffield.ac.uk


## License

This project is licensed under the [NAME HERE] License - see the LICENSE.md file for details

## Acknowledgments

Inspiration, code snippets, etc.
* [awesome-readme](https://github.com/matiassingers/awesome-readme)
* [PurpleBooth](https://gist.github.com/PurpleBooth/109311bb0361f32d87a2)
* [dbader](https://github.com/dbader/readme-template)
* [zenorocha](https://gist.github.com/zenorocha/4526327)
* [fvcproductions](https://gist.github.com/fvcproductions/1bfc2d4aecb01a834b46)


## References
[1] Violeta LP, Ma D, Huang WC, Toda T. Intermediate fine-tuning using imperfect synthetic speech for improving electrolaryngeal speech recognition. In ICASSP 2023-2023 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP) 2023 Jun 4 (pp. 1-5). IEEE.