# MNIST Classifier 

Run a neural network in the browser with javascript by converting a PyTorch model into the [ONNX](https://onnx.ai/) format. We then load the ONNX model in a web-based application (see [here](https://github.com/mcpimentel/react_mnist_classifier)).

We use a simple use-case for recognising handwritten digits! Here, we also attempt to estimate the **uncertainty of the estimations** in order to demonstrate its usefulness (hopefully). We use a PyTorch implementation of Dropout Uncertainty on MNIST. The experiment setting is based on [Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning](https://arxiv.org/pdf/1506.02142.pdf) (i.e., at 5.2 Model Uncertainty in Classification Tasks).

For this demonstration, we train two models based on [LeNet](https://www.paperswithcode.com/method/lenet):

* Lenet standard 

* Lenet dropout


## Demo app 

Check the demo of the [app](https://mcpimentel.github.io/react_mnist_classifier/).


## Code

For the web-based application, check the sibbling repo [https://github.com/mcpimentel/react_mnist_classifier](https://github.com/mcpimentel/react_mnist_classifier).

### Setup

Make sure Pytorch is installed

```
conda install pytorch torchvision -c pytorch
```

and clone this repository

```
git clone https://github.com/mcpimentel/mnist_classifier.git
```

### Usage

Train both models (defined in `models.py`) using 

```
python main.py --mode 0
```

and test the models using

```
python main.py --mode 1
```

Trained models are stored under `checkpoints`.

Finally, convert the trained PyTorch model into ONNX (I found [this tutorial](https://github.com/elliotwaite/pytorch-to-javascript-with-onnx-js) extremely useful; the web app is based on it).

```
python convert_to_onnx.py
```

The PyTorch model description used to generate the ONNX models are defined in `models_inference.py`. 

#### Notes:

* the stable version of ONNX does not support the usage of running inference in *trainingMode* (yet!), which is used for estimating the estimations' uncertainty; given this, we implemented a manual dropout strategy for inference (including a manual random number generator). Hopefully, in the future, this conversion can be easier, and won't need this number of workarounds.

* onnx *opset_version:12* allows the usage of *trainingMode*, but, it is not stable; from a quick look, some operators are a bit flaky. 


#### Debug demo 

* `debug_demo`: a *debug test* to see if the generated ONNX model works. If you are using VSCode, you can use the [Live Server](https://marketplace.visualstudio.com/items?itemName=ritwickdey.LiveServer) extension.
