## Handwritten digits classificator

This project shows implementation of image classificator using tensorflow. Classificator is implemented in a browser with tensorflow.js.

## Prerequisites

Python 3
Tensorflow 2
tensorflowjs_converter

## Running

To learn the model run:

```bash
$ python learn.py
```

To export a model to tensorflow.js:

```bash
$ tensorflowjs_converter --input_format keras model.h5 .\\webpage\\
```
