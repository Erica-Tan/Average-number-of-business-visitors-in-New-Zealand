# Average number of business visitors in New Zealand, 1998-2010

Predicted the average number of business visitors in New Zealand. You can download the dataset in the `data` folder.

### Install the requirements
 
* Install the requirements using `pip install -r requirements.txt`.

### Usage

* Run `preprocess.py` to create the `dataset` and `test` datasets, and create some plot to visulize the data.
    * This will create `dataset.csv` and `test.csv` in the `processed` folder.
* Run `build_model.py`.
    * This will build 3 models including baseline Model, ARIMA Model, ARIMA model (with log transform).
    * It will use Grid Search to find the best parameters.
* Run `predict.py`.
    * This will use the best model to predite the unseen data.
