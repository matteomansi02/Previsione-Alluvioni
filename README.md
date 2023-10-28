# Previsione-Alluvioni
## Introduction
Previsione Alluvioni (Flood Forecasting) is here performed using LSTM neural networks, which are commonly used for predictions of time series data.
The river considered here is the Astico river in the Veneto region of Italy.
The model takes as input past and future meteorological forcings (precipitation, humidity, temperature and solar radiation) and past stage data and predicts the stage 
level of the river at the Lugo di Vicenza measuring station. The Protezione Civile, the italian government body in charge of managing natural disasters, defines for each measuring station stage 
threshold values above which warnings are issued to local authorities and the population. In this work, the yellow level (minor danger) and the orange level (serious danger) are considered,
while the red level (extreme danger) is omitted because of lack of data. In fact, the data is only available for 13 years, from 2010 to 2022 included.

## Data and Files
Data is in .csv format. They are already split in 4 folds for Cross Validation. The numbers before the 's' character indicate which years were included. After 
the 's', are the weather station whose data was included. For example, K10_16_21_s687281.csv contains data from 2010, 2016 and 2021, using the data from station with ID
68, 72, 81. A complete map of stations in Veneto region can be found at https://www.ambienteveneto.it/datiorari/. The _index.csv files are used only to keep track of the timesteps of the data.
loader.py contains the pytorch loader that organizes training and test data into batches.
data.py prepares the data in the appropriate format for the LSTM (3D tensor).
model.py defines the model architecture. It consist of a double encoder whose final context vector is passed onto a decoder. This architecture is used to accomodate different lengths of output.
metrics.py defines the metrics used.
main.py runs everything

## Output
For each K-fold, the observed stage values are plotted against the predicted values. Two lists called 'listone' and 'matrice' are returned. Listone contains the NSE and the contingency matrix for 
orange level events for each Fold, while 'matrice' contains the contingency matrix of yellow and orange events together.

## Usage
The only libraries necessary are pytorch, numpy, pandas, pyplot and the standardscaler function from Sklearn. The various training parameters can be modified in the main.py file

