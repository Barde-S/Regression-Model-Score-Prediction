# Import libraries
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression 
from sklearn.metrics import mean_absolute_error as mae, mean_squared_error as mse, r2_score
from sklearn.model_selection import cross_val_score, KFold
import streamlit as st
st.set_page_config(page_title='Score Prediction', layout='wide')







# Page header and information
st.subheader('Regression Model')
st.write('© The Sparks Foundation')
st.title('Score Prediction Based on Hours of Study')

# Introduction and project overview
st.write('''
The goal of this project is to predict student scores based on the number of hours they study. By analyzing the relationship between study hours and scores, I aim to develop a predictive model that can accurately estimate a student's performance.

The dataset used for this analysis consists of information on students' study hours and corresponding scores. Our objective is to explore the dataset, perform exploratory data analysis (EDA), and develop a predictive model.

During the EDA phase, we will examine the distribution of study hours and scores, identify any outliers or missing values, and explore the correlation between these variables. I will visualize the data using various plots and charts to gain insights into the patterns and trends present.

Next, I will preprocess the data by cleaning any inconsistencies or errors, handling missing values, and encoding categorical variables if necessary. Feature engineering techniques may be applied to create additional relevant features that can enhance the predictive power of the model.

After data preprocessing, I will split the dataset into training and testing sets. I will train different machine learning algorithms, including linear regression, decision trees, random forests, or support vector machines, among others. I will compare the performance of these models using appropriate evaluation metrics and select the best-performing one.

Once the model is trained and evaluated, I will apply it to the testing set to assess its predictive accuracy. I will measure its performance using metrics such as mean absolute error (MAE), root mean squared error (RMSE), or R-squared value. This evaluation will help determine the model's ability to generalize to unseen data accurately.

Finally, I will interpret the results, analyze the significance of study hours on scores, and provide recommendations or insights based on the model's findings.

In summary, this project aims to develop a predictive model to estimate student scores based on the number of study hours. By leveraging EDA techniques and machine learning algorithms, I will uncover patterns and build a robust model that can provide valuable insights and predictions in the field of education.
''')

st.subheader('Let\'s import all libraries needed')
st.markdown('<hr>', unsafe_allow_html=True)
a1='''
import streamlit as st\n
import pandas as pd\n
import matplotlib.pyplot as plt\n
from sklearn.model_selection import train_test_split\n
from sklearn.linear_model import LinearRegression\n
from sklearn.metrics import mean_absolute_error as mae, mean_squared_error as mse, r2_score\n
from sklearn.model_selection import cross_val_score, KFold
'''
st.code(a1, language='python')
st.markdown('<hr>', unsafe_allow_html=True)
'\n'
'\n'

# Loading and reading dataset
st.subheader('Loading and Reading dataset')
st.markdown('<hr>', unsafe_allow_html=True)
url = "http://bit.ly/w-data"
df = pd.read_csv(url)
a2 = '''url = "http://bit.ly/w-data" 
\n
df = pd.read_csv(url) 
'''
st.code(a2, language='python')
'\n'
st.subheader('Let\'s see the first 5 rows of the Dataset')

st.markdown('<hr>', unsafe_allow_html=True)
st.code('df.head()')
st.write(df.head())

# Exploratory Data Analysis (EDA)
st.subheader('Exploratory Data Analysis (EDA)')
st.write('''Let\'s see how both time and score are distributed''')


a4 = '''df.hist(figsize=(3,1))\n
plt.show()'''
st.code(a4, language='python')
df.hist(figsize=(3,1))
st.pyplot(plt)
'\n'
st.write('''You can see that the hours are almost equaly distributed. However people studied between 2-3 hours most.\n
While for scores, people mostly got scores between 25-30''')
 
code='''df.plot(x='Hours', y='Scores', style='o')  
plt.title('Hours vs Percentage')  
plt.xlabel('Hours Studied')  
plt.ylabel('Percentage Score')  
plt.show()'''
st.code(code, language='python')

df.plot(x='Hours', y='Scores', style='o')  
plt.title('Hours vs Percentage')  
plt.xlabel('Hours Studied')  
plt.ylabel('Percentage Score')  
plt.tight_layout()
st.pyplot(plt)
plt.close()
plt.close()

# DataFrame information

st.subheader('DataFrame Information')
st.code('df.shape', language='python')
st.write('Shape:', df.shape)  # Display the shape of the DataFrame
'\n'
'\n'
# Display the DataFrame information in Streamlit
st.write('Columns Data Type')
st.code('df.dtype', language='python')
st.write(df.dtypes)
'\n'
'\n'
st.code('df.isna().sum()', language='python')
st.write(df.isna().sum())
'\n'
st.write('The are no null values across the dataset')
'\n'
st.write('Summarty Statistics of the DataFrame')
st.code('df.describe()', language='python')
st.write(df.describe())
'\n'
'\n'
'\n'
st.write('''Based on the provided dataset, which consists of 25 observations, we can extract some key statistics:

1. Hours:
* Count: There are 25 data points for the "Hours" variable, indicating that we have a complete dataset.
* Mean: The average number of hours studied is approximately 5.012 hours per observation.
* Standard Deviation: The standard deviation is 2.525094, representing the variability or dispersion of the "Hours" data points around the mean.
* Minimum: The minimum value for the "Hours" variable is 1.1, indicating that there is at least one observation with the lowest number of hours studied.
* 25th Percentile: 25% of the data points have a value of 2.7 hours or less, indicating that a quarter of the observations studied for a relatively short period.
* 50th Percentile (Median): The median value for the "Hours" variable is 4.8, implying that 50% of the data points have this value or lower.
* 75th Percentile: 75% of the data points have a value of 7.4 hours or less, indicating that a significant portion of the - observations studied for 7.4 hours or fewer.
* Maximum: The highest value for the "Hours" variable is 9.2, indicating that at least one observation studied for the longest period.
'\n
2. Scores:
* Count: There are 25 data points for the "Scores" variable, indicating that it aligns with the "Hours" variable in terms of completeness.
* Mean: The average score achieved is approximately 51.48, indicating the mean performance across the observations.
* Standard Deviation: The standard deviation for the "Scores" variable is 25.286887, representing the variability or dispersion of the scores around the mean.
* Minimum: The lowest score achieved is 17, indicating that at least one observation had the poorest performance.
* 25th Percentile: 25% of the data points achieved a score of 30 or lower, indicating that a quarter of the observations had relatively lower scores.
* 50th Percentile (Median): The median score is 47, indicating that 50% of the data points have this value or lower.
* 75th Percentile: 75% of the data points achieved a score of 75 or lower, indicating that a significant portion of the observations had scores of 75 or below.
* Maximum: The highest score achieved is 95, indicating that at least one observation had the highest performance.''')
'\n'
'\n'
'\n'
st.subheader('Features Creation')
st.write('''
Machine learning feature creation is the process of transforming raw data into meaningful features that can be used as inputs (X) for training a machine learning model and predicting the target variable (y). The quality and relevance of features play a crucial role in the performance and accuracy of a machine learning model.
Feature creation involves extracting, selecting, and transforming variables from the available dataset to capture relevant information and patterns that can improve the model's predictive power. It requires domain knowledge, understanding of the problem at hand, and careful analysis of the data.''')

a5 = '''X = df.Hours.values \n
y = df.Scores.values \n
X = X.reshape(-1,1)\n
display('Shape of X:', X.shape)\n
display('Shape of y:', X.shape)'''
st.code(a5, language='python')

X = df.Hours.values
y = df.Scores.values
X = X.reshape(-1,1)



st.write('Shape of X:', X.shape)
st.write('Shape of y:', y.shape)

st.markdown('<hr>', unsafe_allow_html=True)

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error as mae, mean_squared_error as mse, r2_score
from sklearn.model_selection import cross_val_score, KFold

st.subheader('Data Splitting')
st.code('X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)', language='python')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

'\n'
st.subheader('Model Training and Prediction')
'\n'
st.code('''reg = LinearRegression()\n
reg.fit(X_train,y_train)\n
predicts = reg.predict(X_test)''', language='python')

reg = LinearRegression()
reg.fit(X_train,y_train)
predicts = reg.predict(X_test)
'\n'

st.subheader('Let\'s see the Predictions on the Test Dataset (X_test)')
predicts
st.markdown('<hr>', unsafe_allow_html=True)
st.subheader('Reshaping of X array then Assign it to a New Variable X_score')
st.code('X_score = X.reshape(-1, 1)', language='python')


X_score = X.reshape(-1, 1)

st.subheader('''Let's create a scatter plot of the data points, then fit a regression line to the data, and visualizes the relationship between hours and scores.''')
st.code('''plt.scatter(X, y, color='blue', label='Student performance per hours spent')

# Predicted values using the fitted model
y_pred = reg.predict(X)

# Plotting the regression line
plt.plot(X, y_pred, color='red', linewidth=2, label='Regression Line')

# Set labels and title
plt.xlabel('Hours')
plt.ylabel('Score')
plt.title('Regression Line')

# Show legend
plt.legend()

# Show the plot
plt.show()''', language='python')


# Create a scatter plot
plt.figure(figsize=[6,4])
plt.scatter(X, y, color='blue', label='Student performance per hours spent')

# Predicted values using the fitted model
y_pred = reg.predict(X)

# Plotting the regression line
plt.plot(X, y_pred, color='red', linewidth=2, label='Regression Line')

# Set labels and title
plt.xlabel('Hours')
plt.ylabel('Score')
plt.title('Regression Line')

# Show legend
plt.legend()



# Display the plot in Streamlit
st.pyplot(plt)
st.write('''This alignment and clustering around the regression line indicate that as the independent variable (X) increases, the dependent variable (y) also tends to increase or decrease consistently. The data points being close to the regression line suggest that the model is able to capture the underlying trend in the data.

This alignment and clustering can be interpreted as a high degree of correlation between the variables, and the regression line can be used to make predictions or estimate the value of the dependent variable based on the independent variable.''')

st.markdown('<hr>', unsafe_allow_html=True)

st.subheader('Comparing Between Actual Score and Predicted Score')
st.code('''lin = pd.DataFrame({'Actual': y_test, 'Predicted': predicts})\n
print(lin)''', language='python')
lin = pd.DataFrame({'Actual': y_test, 'Predicted': predicts})
lin
st.write('''The provided code creates a pandas DataFrame called 'lin' that contains two columns: 'Actual' and 'Predicted'. 

- The 'Actual' column represents the actual values of the dependent variable (y) from the test dataset.
- The 'Predicted' column contains the predicted values of the dependent variable (y) obtained from the linear regression model.

The DataFrame is then printed, displaying the actual and predicted values side by side. This allows for easy comparison and evaluation of the model's performance. By examining the DataFrame, you can assess how well the model's predictions align with the actual values.''')
st.markdown('<hr>', unsafe_allow_html=True)

st.subheader('''Model Performance By Comparing Between the Actual and Predicted Value''')
st.markdown('<hr>', unsafe_allow_html=True)

st.write('''Based on this information, we can evaluate the performance of the model in predicting the scores.

Observation 1 shows that the model predicted a score of 83.188141, which is slightly higher than the actual score of 81. This suggests that the model performed relatively well in this case, accurately predicting a score close to the actual value.

Observation 2 indicates that the model predicted a score of 27.032088, whereas the actual score was 30. The model underestimated the score in this instance, indicating a slight discrepancy between the predicted and actual values.

Observation 3 also reveals a similar pattern, with the model predicting a score of 27.032088, whereas the actual score was 21. Again, the model underestimated the score, resulting in a difference between the predicted and actual values.

Observation 4 demonstrates that the model predicted a score of 69.633232, which is relatively close to the actual score of 76. The model's prediction in this case is reasonably accurate, although there is still a slight difference between the predicted and actual values.

Observation 5 shows that the model predicted a score of 59.951153, while the actual score was 62. The model's prediction is quite close to the actual value, indicating a good level of accuracy.

Overall, based on this limited sample, it appears that the model is relatively effective in predicting scores.''')

st.markdown('<hr>', unsafe_allow_html=True)
st.subheader('Predicting Score For Studying For 9.25 Hours')

a6 = '''hours = np.array([[9.25]])
predictions = reg.predict(hours)
print("Number of Hours = {}".format(hours[0][0]))
print("Predicted Score = {}".format(predictions[0]))'''

hours = np.array([[9.25]])
predictions = reg.predict(hours)
st.write("Number of Hours = {}".format(hours[0][0]))
st.write("Predicted Score = {}".format(predictions[0]))

st.write('The model predicted that a student who studied for 9.25 will have an approximate score of 92.4')
st.markdown('<hr>', unsafe_allow_html=True)

st.subheader('Evaluation of Model Performance')
import streamlit as st

st.markdown("<b><u>Cross Validation</u></b>", unsafe_allow_html=True)


st.write('Cross-validation is a vital approach to evaluating a model. It maximizes the amount of data that is available to the model, as the model is not only trained but also tested on all of the available data.')


code2 = '''# Create a KFold object
kfold = KFold(n_splits=6, shuffle=True, random_state=5)

# Perform cross-validation
cv_scores = cross_val_score(reg, X, y, cv=kfold)

# Calculate and print average score and standard deviation
avg_score = cv_scores.mean()
std_dev = cv_scores.std()'''

st.code(code2, language='python')

# Create a KFold object
kfold = KFold(n_splits=6, shuffle=True, random_state=5)

# Perform cross-validation
cv_scores = cross_val_score(reg, X, y, cv=kfold)

# Calculate and print average score and standard deviation
avg_score = cv_scores.mean()
std_dev = cv_scores.std()

st.write('''Cross-Validation Scores: [0.90005467, 0.94662445, 0.7556832,  0.93657642, 0.95833353, 0.93102917]
- Average Score: 0.9047169070303717
- Standard Deviation: 0.06900619783075516
- The model achieved an average score of 0.9047169070303717 with a standard deviation of 0.06900619783075516
- This indicates that the model explains the variation in the target variable reasonably well.''')
st.markdown('<hr>', unsafe_allow_html=True)


st.subheader('RMSE. MSE, MAE and R2 Score')

code3 = '''Model_Performance4= { 
    
                      'Evaluating the model':
                    
                         {"Root mean squared error": (np.sqrt(mse(y_test,predicts))),
                        "Mean squared error": (mse(y_test,predicts)),
                        "Mean absolute error": (mae(y_test,predicts)),
                        "R squared": (r2_score(y_test,predicts))}
                        
                    }

# create dataframe from dictionary
Model_Performance4 = pd.DataFrame(data=Model_Performance4)
Model_Performance4'''
st.code(code3, language='python')
Model_Performance4= { 
    
                      'Evaluating the model':
                    
                         {"Root mean squared error": (np.sqrt(mse(y_test,predicts))),
                        "Mean squared error": (mse(y_test,predicts)),
                        "Mean absolute error": (mae(y_test,predicts)),
                        "R squared": (r2_score(y_test,predicts))}
                        
                    }

# create dataframe from dictionary
Model_Performance4 = pd.DataFrame(data=Model_Performance4)
Model_Performance4

st.markdown('<hr>', unsafe_allow_html=True)

st.write('''Based on the output, we can draw the following observations:

- Mean Absolute Error (MAE): The MAE value is 3.920751. This metric represents the average absolute difference between the predicted values and the actual values. A lower MAE indicates better model performance in terms of accuracy and precision.

- Mean Squared Error (MSE): The MSE value is 18.943212. This metric measures the average squared difference between the predicted values and the actual values. It gives more weight to larger errors compared to MAE. Similar to MAE, a lower MSE value indicates better model performance.

- R-squared (R^2): The R-squared value is 0.967806. This metric provides an indication of how well the model fits the observed data. It represents the proportion of the variance in the dependent variable that is predictable from the independent variables. A higher R-squared value closer to 1 indicates that the model explains a large portion of the variance in the data, suggesting a good fit.

- Root Mean Squared Error (RMSE): The RMSE value is 4.352380. This metric is the square root of the MSE and represents the standard deviation of the residuals. It provides a more interpretable measure of the average prediction error. A lower RMSE value indicates better model performance in terms of accuracy.

Overall, the model performance appears to be quite promising. The low values for MAE, MSE, and RMSE indicate that the model's predictions are relatively close to the actual values. Additionally, the high R-squared value suggests that a significant portion of the variance in the dependent variable is captured by the model. However, it's important to compare these metrics with the context of the problem and other models to fully assess the performance.''')
st.markdown('<hr>', unsafe_allow_html=True)


code5 = '''# Get user input for hours spent
hours_spent = float(input("Enter the number of hours spent: "))

# Convert the input to a numpy array
hours_spent_arr = np.array([[hours_spent]])

# Make predictions
predictions = reg.predict(hours_spent_arr)

# Print the results
print("Number of Hours = {}".format(hours_spent_arr[0][0]))
print("Predicted Score = {}".format(predictions[0]))'''


st.subheader('Predict Your Own Score')
# Get user input for hours spent
hours_spent = st.sidebar.number_input("Enter the number of hours spent", step=0.1)

# Convert the input to a numpy array
hours_spent_arr = np.array([[hours_spent]])

# Make predictions
predictions = reg.predict(hours_spent_arr)

# Display the results
st.write("Number of Hours =", hours_spent_arr[0][0])
st.write("Predicted Score =", predictions[0])



st.write('<p style="text-align: right;">By Shuaibu Sani Barde</p>', unsafe_allow_html=True)
