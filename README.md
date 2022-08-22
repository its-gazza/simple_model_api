# Simple API server for a Sklearn model

## Usage

### Train model

Call:

```sh
    python3 train.py
```

This is a very simple script that trains a KNN model with the `iris` dataset

### Start server

Call:

```sh 
    python3 api.py
```

### Prediction

Call:

```sh
    curl -X POST -d '[2, 3, 4, 5]' http://localhost:8080/predict -H "Content-Type: application/json"
```