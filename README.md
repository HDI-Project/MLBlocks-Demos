# MLBLocks-Demos

Some example MLPipelines and code to test them on sample datasets

## Installation

Just run

```
pip install -r requirements.txt
```

This will pull and install MLBlocks from the githup repository.

No package installation command is required, as all the scripts are independent.

### Special requirements

In order to be able to run the audio pipeline, `ffmpeg` needs to be installed:

```
sudo apt-get install ffmpeg
```

## Usage

The simplest way to run an example is by runnnig each script inside the
examples folder independently:

```
python examples/path/to/the/example.py
```

However, the script `run_example.py` provides a more convenient way to run them, with
the possibility to pass custom arguments and store the results for later review.

It accepts the following parameters:

* **example**: It can be module path, or the FQN of a module or just the example name.
               If an example name is given, some default arguments are passed to it.
* **--output, -o**: path to a folder where the results of the test will be stored as a JSON file.
* **extra arguments**: Any extra argument written after the example name as `key=value` is
                       passed to the example `run` method.


These are some valid examples:

```
python run_example.py examples/image/simple_cnn_classifier.py
python run_example.py examples.image.simple_cnn_classifier
python run_example.py simple_cnn_classifier test_size=0.3 epochs=12
python run_example.py simple_cnn_classifier -o test_results
```
