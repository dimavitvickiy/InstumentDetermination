# Instrument Determination
Instrument determination using neural networks.
Available instruments:
- basson
- violin
- double bass
- trombone
- tuba
- flute
- cello
- french horn

## Requirements
python == 3.6+

### Install requirements
`virtualenv .venv -p=python3.6`
`pip install -r requirements.txt`

### Predict instrument
`python predict --filename=path_to_file`

### Fetch features
1. create dir `samples`
2. add samples of soundtracks to sample dir in appropriate folder
(example, cello soundtrack samples must be in dir `samples/cello`)
3. `run python extract_all_features.py`

### Build features plot
`python instrument_features.py`

### Learn neural network
`python neural_network_tensorflow.py`

### Note: neural network's classifier holds in `model.pickle` file

## Results: 

![confusion matrix](https://github.com/dimavitvickiy/InstumentDetermination/blob/master/instruments.png?raw=true)
![confusion matrix](https://github.com/dimavitvickiy/InstumentDetermination/blob/master/prediction.png?raw=true)
