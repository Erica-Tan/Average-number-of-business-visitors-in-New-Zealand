# Average number of business visitors in New Zealand, 1998-2010

Forecasted the average number of business visitors in New Zealand. You can download the dataset in the `data` folder.

### Install the requirements
 
* Install the requirements using `pip install -r requirements.txt`.

### Usage

* Run `data_preposessing.py` to create the `train` and `test` datasets, and create some plot to visulize the data.
    * This will create `train.csv` and `test.csv` in the `processed` folder.
* Run `baseline_model.py`.
	* This will run Persistence forecast model across the training set, and evaluate the model on test set.
* Run `ARIMA_model.py`.
    * This will build 2 models including ARIMA Model, ARIMA model (with log transform).
    * It will use Grid Search to find the best parameters.
    * It will save the final model called `arima_model.pkl` to the output folder.
* Run `LSTM_model.py`.
    * This will run Long short-term memory Network across the training set, and evaluate the model on test set.
    * It will save the final model called `lstm_model.json` and `lstm_model.h5` to the output folder.
