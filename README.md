# meta-darts


## Step 1.

```python
python train_search.py --help
```
This is search process for architectures

## Step 2.

Copy best architecture genotypes, paste and name well in genotypes.py


## Step 3.

```python
python train.py --arch NAME
```
This is training process for specific architecture on cifar10

here NAME refers to the achitecture genotype you defined in step 2, e.g. DARTS_V1


## Step 4.

```python
python train_imagenet.py --arch NAME
```

This is training process for specific architecture on imagenet
