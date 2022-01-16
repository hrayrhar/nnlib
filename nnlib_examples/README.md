# Some examples of using the `nnlib` library

Training a 4-layer CNN on MNIST.
```bash
python -um examples.scripts.train_classifier -c examples/configs/mnist-4layer-cnn.json -d cuda:0 -b 128 -e 100 \
    -s 10 -v 10 -l logs/mnist-4layer-cnn -D mnist 
```

Fine-tuning a pretrained ResNet-50 on CIFAR-10.
```bash
python -um examples.scripts.train_classifier -c examples/configs/pretrained-resnet50-cifar10.json -d cuda:0 \
  -b 32 -e 100 -s 10 -v 10 -l logs/pretrained-resnet50-cifar10 -D cifar10 --resize_to_imagenet \
  --optimizer sgd --lr 0.003 --momentum 0.9 --weight_decay 0.001
```