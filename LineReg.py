
import pandas as pd            
import matplotlib.pyplot as plt  
from sklearn.model_selection import train_test_split  
from sklearn.linear_model import LinearRegression     
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score 


data = pd.read_csv("salary_data.csv")


print("First 5 rows of the dataset:")
print(data.head())


X = data[['YearsExperience']] 
y = data['Salary']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

model = LinearRegression()
model.fit(X_train, y_train)


y_pred = model.predict(X_test)


print("\nModel Evaluation Results:")
print("Mean Absolute Error (MAE):", mean_absolute_error(y_test, y_pred))
print("Mean Squared Error (MSE):", mean_squared_error(y_test, y_pred))
print("R-squared (R²):", r2_score(y_test, y_pred))  


plt.scatter(X_test, y_test, color='blue', label='Actual Salary')   
plt.plot(X_test, y_pred, color='red', label='Predicted Salary')    
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.title("Simple Linear Regression: Experience vs Salary")
plt.legend()
plt.grid(True)
plt.show()

slope = model.coef_[0]       
intercept = model.intercept_ 
print("\n The formula the model learned:")
print(f"   Salary = {slope:.2f} * YearsExperience + {intercept:.2f}")
