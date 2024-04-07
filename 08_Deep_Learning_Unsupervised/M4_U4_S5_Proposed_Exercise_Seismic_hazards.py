# 5. PROPOSED EXERCISE
# For this proposed exercise we will use the dataset available in
# http://odds.cs.stonybrook.edu/seismic-dataset/
#
# This dataset contains geological information linked to seismic hazards. The data set is labeled for
# abnormal cases (when there has been an earthquake) versus non-abnormal cases (there has
# been no earthquake).
# This dataset is in a different format than usual, in .arff. The way to read it and pass it on to a
# Pandas dataframe is as follows.

from scipy.io import arff
data = arff.loadarff('seismic-bumps.arff')
df = pd.DataFrame(data[0])
df['class'] = df['class'].astype(int)
df['seismic'] = df['seismic'].str.decode("utf-8")
df['seismoacoustic'] = df['seismoacoustic'].str.decode("utf-8")
df['shift'] = df['shift'].str.decode("utf-8")
df['ghazard'] = df['ghazard'].str.decode("utf-8")

# For the exercise, the following are requested:
# ▪ Describe and present the dataset and variables involved.
# ▪ Present a small exploratory analysis of the data.
# ▪ Detect anomalies in the dataset using either an AE or a SOM. For both cases, visually
# choose the threshold value (for MID or for reconstruction error) and display the confusion
# matrix for model predictions. Other plots that are considered relevant can be displayed.


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, f1_score
from scipy.io import arff
def autoencoder_function(nv, nh, optimizer="adam",loss='mean_squared_error',metrics=['accuracy']):
"""
Main function to define the AE Neural Network.
Parameters
----------
nv : TYPE
DESCRIPTION.
nh : TYPE
DESCRIPTION.
optimizer : TYPE, optional
DESCRIPTION. The default is "adam".
loss : TYPE, optional
DESCRIPTION. The default is 'mean_squared_error'.
metrics : TYPE, optional
DESCRIPTION. The default is ['accuracy'].
Returns
-------
autoencoder : TYPE
DESCRIPTION.
"""
# Define input
input_layer = Input(shape=(nv,))
# Encoding
encoder = Dense(nh, activation='relu',
activity_regularizer=regularizers.l1(10e-5))(input_layer)
# Decoding
decoder = Dense(nv, activation='sigmoid')(encoder)
# Model
autoencoder = Model(inputs=input_layer, outputs=decoder)
autoencoder.compile(optimizer=optimizer, loss=loss, metrics=metrics)
return autoencoder
1. Description of the dataset
The dataset used contains information about seismic problems linked to certain conditions with
19 input attributes. In this way, there are two types of outputs: Hazard to indicate that those
input variables can indicate a seismic risk, and No Hazard to the contrary. The problem is
unbalanced as there are many more records of stable conditions than of risky conditions. The
problem provides the binary output variable that distinguishes between the two scenarios.
However, in many cases this information will not be known, and that is where unpervised
anomaly detection can help predict seismic risks by being associated with combinations of
values in input variables other than the usual scenarios.
1. seismic: result of shift seismic hazard assessment in the mine working obtained by the seismic method (a - lack of hazard, b - low hazard, c - high hazard, d - danger state).
2. seismoacoustic: result of shift seismic hazard assessment in the mine working obtained by the seismoacoustic method.
3. shift: information about type of a shift (W - coal-getting, N -preparation shift).
4. genergy: seismic energy recorded within previous shift by the most active geophone (GMax) out of geophones monitoring the longwall;
5. gpuls: a number of pulses recorded within previous shift by GMax;
6. gdenergy: a deviation of energy recorded within previous shift by GMax from average energy recorded during eight previous shifts;
7. gdpuls: a deviation of a number of pulses recorded within previous shift by GMax from average number of pulses recorded during eight previous shifts;
8. ghazard: result of shift seismic hazard assessment in the mine working obtained by the seismoacoustic method based on registration coming form GMax only;
9. nbumps: the number of seismic bumps recorded within previous shift;
10. nbumps2: the number of seismic bumps (in energy range [10^2,10^3)) registered within previous shift;
11. nbumps3: the number of seismic bumps (in energy range [10^3,10^4)) registered within previous shift;
12. nbumps4: the number of seismic bumps (in energy range [10^4,10^5)) registered within previous shift.
13. nbumps5: the number of seismic bumps (in energy range [10^5,10^6)) registered within the last shift.
14. nbumps6: the number of seismic bumps (in energy range [10^6,10^7)) registered within previous shift.
15. nbumps7: the number of seismic bumps (in energy range [10^7,10^8)) registered within previous shift.
16. nbumps89: the number of seismic bumps (in energy range [10^8,10^10)) registered within previous shift.
17. energy: total energy of seismic bumps registered within previous shift.
18. maxenergy: the maximum energy of the seismic bumps registered within previous shift.
19. class: the decision attribute - '1' means that high energy seismic bump occurred in the next shift
('hazardous state'), '0' means that no high energy seismic bumps occurred in the next shift
('non-hazardous state').

2. Brief exploratory analysis
First, we load the data (in .arff) and we do a decoding of the columns in text format to be able to
read their values.
data = arff.loadarff('seismic-bumps.arff')
df = pd.DataFrame(data[0])
df['class'] = df['class'].astype(int)
df['seismic'] = df['seismic'].str.decode("utf-8")
df['seismoacoustic'] = df['seismoacoustic'].str.decode("utf-8")
df['shift'] = df['shift'].str.decode("utf-8")
df['ghazard'] = df['ghazard'].str.decode("utf-8")
As an example, the variables for one of the records are:
We continue to make a LabelEncoding
plus
OneHotEncoding
of non-ordinal categorical
variables.
df = pd.get_dummies(df, drop_first=True)
Next, we analyze the distribution of the output variable. As we can see, there is a clear
imbalance, as expected. Approximately 93% of records correspond to scenarios without seismic
risks.
Subsequently, we delete the n orrepresentative variables because they have constant values.
To do this, we make sure that your typical deviation is greater than 0, and if not, we eliminate
them.
# Eliminate low variance cols
cols_keep = [column for column in list(df.columns) if df[column].std()>0]
df = df[cols_keep]
We continue in the EDA with a bivariate analysis for numeric variables, where we include
dependency with the output variable.
# Plot correlation
import matplotlib.pyplot as plt
import seaborn as sns
list_no = ['seismic_b', 'seismoacoustic_b','seismoacoustic_c', 'shift_W', 'ghazard_b', 'ghazard_c']
cols = [x for x in list(df.columns) if x not in list_no]
f,ax = plt.subplots(figsize=(20,20))
df_plot = df[cols] # columnas que no esten en obj_df
sns.heatmap(df_plot.corr(method='spearman'),annot=True,fmt=".1f",linewidths=1,ax=ax)
plt.show()
As we can see, there seems to be no variables with a clear dependency on the output variable.
Some even seem to have a correlation of 0. Other variables show how there is a linear
dependency on between them. These are removed from the dataset. That includes: maxenergy
and nbumps. The others, although they may have high correlations, we leave them for the time
being. Similarly, even if they have a correlation of zero with respect to the output, those
variables are also left because they are correlated with others that do have dependence on the
output.
To see the dependence of categorical variables with the output variable we apply an ANOVA
hypothesis contrast.
import scipy.stats as stats
print("ANOVA for seismic_b")
print(stats.f_oneway(df[df["seismic_b"]==0]["class"],
df[df["seismic_b"]==1]["class"]))
print()
print("ANOVA for seismoacoustic_b")
print(stats.f_oneway(df[df["seismoacoustic_b"]==0]["class"],
df[df["seismoacoustic_b"]==1]["class"]))
print()
print("ANOVA for seismoacoustic_c")
print(stats.f_oneway(df[df["seismoacoustic_c"]==0]["class"],
df[df["seismoacoustic_c"]==1]["class"]))
print()
print("ANOVA for shift_W")
print(stats.f_oneway(df[df["shift_W"]==0]["class"],
df[df["shift_W"]==1]["class"]))
print()
print("ANOVA for ghazard_b")
print(stats.f_oneway(df[df["ghazard_b"]==0]["class"],
df[df["ghazard_b"]==1]["class"]))
print()
print("ANOVA for ghazard_c")
print(stats.f_oneway(df[df["ghazard_c"]==0]["class"],
df[df["ghazard_c"]==1]["class"]))
The output obtained is:
ANOVA for seismic_b
F_onewayResult(statistic=21.356901872363814, pvalue=3.999798007997421e-06)
ANOVA for seismoacoustic_b
F_onewayResult(statistic=0.26030552095122494, pvalue=0.6099539459472678)
ANOVA for seismoacoustic_c
F_onewayResult(statistic=0.008604018540960165, pvalue=0.9261031344898955)
ANOVA for shift_W
F_onewayResult(statistic=53.195573029878325, pvalue=3.9966222710548486e-13)
ANOVA for ghazard_b
F_onewayResult(statistic=0.0002314181198416004, pvalue=0.9878638779704311)
ANOVA for ghazard_c
F_onewayResult(statistic=2.1376059696220433, pvalue=0.14384772674207144)
In this way, with a reference of 0.05, we only see significant differences in the means of the
groups between the variables seismic_b and shift_W.
In the case of a supervised classic problem, non-relevant variables would be removed to
improve the predictive power of the model.

However, as we are going to consider an unsupervised anomaly analysis, let's assume that
there is no such a priori information of the output category, and therefore we do not have
information to delete variables based on their dependency on output. We will only delete them
based on the information between them. For this reason, we exclusively eliminate highly
correlated variables.
Along with this, in the next step we normalize the variables so that all of them are in the same
range of values. As we seek to detect anomalies, we will apply a normalization, since this
technique is especially sensitive to abnormally low or high values in variables. This is because
these values will define the maximum and minimum references (1 and 0) for which the rest of
the values will be adjusted.
# Delete columns
df = df.drop(columns=['maxenergy', 'nbumps'])
# Normalize
sc = MinMaxScaler(feature_range = (0, 1))
list_columns = list(df.columns)
df = pd.DataFrame(sc.fit_transform(df))
df.columns = list_columns
We end by visually displaying the ratio and data between the classes of the output variable.
# Disable eager_exec
tf.compat.v1.disable_eager_execution()
df_raw = df
# Plot target variable
labels = ['Normal', 'Seismic']
sizes = [len(df_raw[df_raw['class']==0]), len(df_raw[df_raw['class']==1])]
colors = ['lightskyblue','red']
explode = (0, 0.1)
plt.pie(sizes, explode=explode, labels=labels, colors=colors,
autopct='%1.1f%%', shadow=True, startangle=140)
plt.title("Normal vs Seismic")
3. Detection of anomalies
After the processing, preparation and exploration of data, we proceed to the detection of
anomalies using
AutoEncoders. We will do this with both supervised and unsupervised
approximation (in order to define the threshold value of the rebuild error).
First, for supervised approximation, we train the AE using the non-abnormal data.
We define a threshold for the reconstruction and see how the anomaly data is separated from
the non-abnormal data based on the threshold specified for that error.

Metrics such as the confusion matrix or F1 value (especially useful since there is data
imbalance) can also be checked to assess anomaly detection efficiency at the chosen
threshold.
#
========================================================================
=
# 2. Supervised
#
========================================================================
=
# Train/Eval split
X_train, X_test = train_test_split(df_raw, test_size=0.2, random_state=42)
# Train only in non outliers
X_train = X_train[X_train['class'] == 0]
X_train = X_train.drop(['class'], axis=1)
y_test = X_test['class']
X_test = X_test.drop(['class'], axis=1)
X_train = X_train.values
X_test = X_test.values
# Parameters
nv = X_train.shape[1] # visible unists
nh = 32 # hidden units
epochs=50


batch_size=256
# Train AE
model = autoencoder_function(nv, nh,optimizer="adam", loss='mean_squared_error',
metrics=['accuracy'])
history = model.fit(X_train, X_train,
epochs=epochs,
batch_size=batch_size,
shuffle=True)
# Plot metrics
plt_history = history.history
plt.plot(plt_history['loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['validation'], loc='upper right')
# Obtain predictions
threshold = 0.02 # Set threshold for reconstruction error
predictions = model.predict(X_test)
se = np.mean(np.power(X_test - predictions, 2), axis=1)
mse = np.mean(np.power(X_test - predictions, 2), axis=1)
error_df = pd.DataFrame({'reconstruction_error': mse,
'true_class': y_test})


# Visualize [Chart]
groups = error_df.groupby('true_class')
fig, ax = plt.subplots()
for name, group in groups:
ax.plot(group.index, group.reconstruction_error, marker='o', ms=3.5, linestyle='',
label= "Outlier" if name == 1 else "Normal")
ax.hlines(threshold, ax.get_xlim()[0], ax.get_xlim()[1], colors="r", zorder=100, label='Threshold')
ax.legend()
plt.title("Reconstruction error for different classes")
plt.ylabel("Reconstruction error")
plt.xlabel("Data point index")
plt.show()
# Visualize [CM]
y_pred = [1 if e > threshold else 0 for e in error_df.reconstruction_error.values]
conf_matrix = confusion_matrix(error_df.true_class, y_pred)
sns.heatmap(conf_matrix, xticklabels=labels, yticklabels=labels,
annot=True, fmt="d", annot_kws={"fontsize":12})
plt.title("Confusion matrix")
plt.ylabel('True class')
plt.xlabel('Predicted class')
plt.show()
# Other metrics
print("F1 score: ", f1_score(error_df.true_class, y_pred))


F1: 0.178


As we can see, although the defined threshold manages to separate a few abnormal points, the
metrics obtained are not particularly good. By adjusting the threshold, they could be improved.
For example, lowering it to 0.013 the confusion matrix improves to:
Other ways to explore to improve anomaly detection capability would also be by adjusting the
parameters and architecture of the AE used.
Finally, in the unsupervised approximation, the threshold of the reconstruction error should be
chosen visually based on how the data is distributed, as shown below.
#
========================================================================
=
# 3. Unsupervised
#
========================================================================
=


# Train/Eval split
df_input = df_raw.copy().drop(columns=['class']) # No class column
X_train, X_test = train_test_split(df_input, test_size=0.2, random_state=42)
# Parameters
nv = X_train.shape[1] # visible unists
nh = 32 # hidden units
epochs=50
batch_size=256
# Train AE
model = autoencoder_function(nv, nh,optimizer="adam", loss='mean_squared_error',
metrics=['accuracy'])
history = model.fit(X_train, X_train,
epochs=epochs,
batch_size=batch_size,
shuffle=True)
# Obtain predictions
threshold = 0.02 # Set threshold for reconstruction error
predictions = model.predict(X_test)
se = np.mean(np.power(X_test - predictions, 2), axis=1)
mse = np.mean(np.power(X_test - predictions, 2), axis=1)
error_df = pd.DataFrame({'reconstruction_error': mse})


# Visualize [Chart]
fig, ax = plt.subplots()
ax.plot(error_df.index, error_df.reconstruction_error, marker='o', ms=3.5, linestyle='')
ax.hlines(threshold, ax.get_xlim()[0], ax.get_xlim()[1], colors="r", zorder=100, label='Threshold')
ax.legend()
plt.title("Reconstruction error for unlabelled data")
plt.ylabel("Reconstruction error")
plt.xlabel("Data point index")
plt.show()

