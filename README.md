# ez-mlmd
Quick and easy application of machine learning to molecular dynamics trajectories

The structure of protein comparing to its native state is usually estimated by fraction of native contacts:

![qn](demo/eqn-qnative.png)

However, the tolerance of distance fluctuation is set to be uniform for all native contacts, which are composed of different amino acid pairs with different significance of interactions.

Here, a different approach is proposed to determine whether the protein is in its native state based on machine learning of its equilibration trajectories at different temperatures. Specifically, the Cartesian coordinates of heavy atoms in each frame will be used as training sets. All frames simulated at 300K will be labeled as in "native state" (`1`) and structures at 600K will be considered as "unfolded" (`0`). 

Currently only criteria of nativeness is calculated but advanced applications such as feature extraction can be added when necessary, since all the coordinates are retrieved based on current implementation.


Equilibrations at 300K (green) and 600K (pink) will be used for training.

<img src="demo/demo-temp300.gif" width="40%" height="40%" alt="eq300k" align="center" />
<img src="demo/demo-temp600.gif" width="40%" height="40%" alt="eq600k" align="center" />

A steered molecular dynamics (SMD) trajectory is used for prediction. Note that the protein is considered as non-native immediately after the first unfolding event.

<img src="demo/demo-smd.gif" width="40%" height="40%" alt="smd300k" align="center" />

### Example output

```
Using TensorFlow backend.
Instructions for updating:
Colocations handled automatically by placer.
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
dense_1 (Dense)              (None, 56)                73304
_________________________________________________________________
dense_2 (Dense)              (None, 28)                1596
_________________________________________________________________
dense_3 (Dense)              (None, 14)                406
_________________________________________________________________
dense_4 (Dense)              (None, 1)                 15
=================================================================
Total params: 75,321
Trainable params: 75,321
Non-trainable params: 0
_________________________________________________________________
None
---------------
Accuracy = 1.00
---------------
```

<img src="demo/demo-epoch-accu-loss.png" width="60%" height="60%" alt="demo-epoch-accu-loss"/>
<img src="demo/demo-frame-pred-force.png" width="60%" height="60%" alt="demo-frame-pred-force"/>
