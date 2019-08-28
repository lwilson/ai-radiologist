# AI Radiologist

This is an implementation of CheXNet model which is a 121 layer convolutional neural network model(DenseNet121). The model helps to identify one or more thoracic pathologies from the 14 different diagnosis identified in the ChestXray14 frontal chest x-ray training data. Here, we also provide a way to perform distributed training across multiple nodes with the help of horovod.

## Getting Started

Clone the files to a lcoal  directory or mounted directory in case of nauta.  

```
# installing intel optimized tensorflow
# https://software.intel.com/en-us/articles/intel-optimization-for-tensorflow-installation-guide
# tensorflow version -> tensorflow==1.12.0

# download and install the intel optimized tensorflow

wget https://storage.googleapis.com/intel-optimized-tensorflow/tensorflow-1.12.0-cp36-cp36m-linux_x86_64.whl -O ./tensorflow-1.12.0-cp36-cp36m-linux_x86_64.whl

pip install --no-cache-dir --force-reinstall ./tensorflow-1.12.0-*.whl

# install other required python libraries in case of bare metal
pip install --no-cache-dir keras pillow sklearn horovod=0.16.0
```


## General Instructions

Ensure that the ai-radiologist directory is appended to the PYTHONPATH correctly.

This can be done by updating the sys.path in multiclass_horovod_optimizers.py.

```
sys.path.append('/path/to/ai-radiologist')
```

Using Nauta:

```
sys.path.append('/mnt/output/home/ai-radiologist')
```


## Data

The [ChestX-ray14 dataset] (https://nihcc.app.box.com/v/ChestXray-NIHCC) is used for training and evaluation. It consists of 112,120 frontal-view X-ray images of 30,805 unique patients. Each image is labelled with up to 14 different thoracic pathologies.

## Train

Train using 4 nodes and 4 processes each.

Export the required environment variables. This needs to be be udpated in templates values.yaml in case of nauta.

```
export OMP_NUM_THREADS=9
export KMP_BLOCKTIME=0
```

Set APP_DIR, DATA_DIR and MODEL_DIR paths accordingly.

```
mpiexec --hostfile $hostfile -cpus-per-proc 9 --map-by socket --oversubscribe -n 16 -x OMP_NUM_THREADS -x KMP_BLOCKTIME python ${APP_DIR}/multiclass_horovod_optimizers.py --data_dir=${DATA_DIR}/images_all/ --model_dir=${MODEL_DIR} --epochs=15 --train_label=${DATA_DIR} --validation_label=${DATA_DIR}
```

Using Nauta:

Update the paths to multiclass_horovod_optimizers.py, data_dir, train_label, validation_label accordingly.

```
nctl exp submit --name ai-rad-test -t multinode-tf-training-horovod /path/to/multiclass_horovod_optimizers.py -- --data_dir='/mnt/output/home/data/ai-radiologist/images_all/' --epochs=15 --train_label='/mnt/output/home/data/ai-radiologist' --validation_label='/mnt/output/home/data/ai-radiologist'
```



### Eval

Evaluate the model by loading the trained weight file.

Update paths to auc_calculate.py, data_dir, train_label, validation_label, weights_file accordingly.

```
python auc_calculate.py \
--data_dir='/home/srinivas/nauta_bm_tests/rambler_r1.0/data/ai-radiologist/images_all' \
--train_label='/home/srinivas/nauta_bm_tests/rambler_r1.0/data/ai-radiologist' \
--validation_label='/home/srinivas/nauta_bm_tests/rambler_r1.0/data/ai-radiologist' \
--weights_file='/home/srinivas/nauta_bm_tests/models/navsbm_test2_bm-6230-rad-2-4-test1/lr_0.001_bz_16_loss_0.160_epoch_01.h5'
```


Using Nauta:

Update paths to auc_calculate.py, data_dir, train_label, validation_label, weights_file accordingly.

```
nctl exp submit --name ai-rad-auc_calculate -t multinode-tf-training-horovod /path/to/ai-radiologist/auc_calculate.py -- --data_dir='/mnt/output/home/data/ai-radiologist/images_all/' --train_label='/mnt/output/home/data/ai-radiologist' --validation_label='/mnt/output/home/data/ai-radiologist' --weights_file='/mnt/output/home/ai-rad-test/lr_0.001_bz_16_loss_0.206_epoch_01.h5'

```


## Related articles
  https://community.emc.com/community/products/rs_for_ai/blog/2018/08/16/training-an-ai-radiologist-with-distributed-deep-learning 


