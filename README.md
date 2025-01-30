import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
data = pd.read_csv('shopping_data.csv')  # Replace with your dataset file name

# Display basic information about the dataset
print("Dataset Info:")
print(data.info())

# Data Cleaning
data.dropna(inplace=True)  # Remove rows with missing values
data['PurchaseDate'] = pd.to_datetime(data['PurchaseDate'])  # Convert to datetime

# Add new columns for analysis
data['Month'] = data['PurchaseDate'].dt.month
data['Year'] = data['PurchaseDate'].dt.year

# Analyze Monthly Sales Trends
monthly_sales = data.groupby(['Year', 'Month'])['Sales'].sum().reset_index()
plt.figure(figsize=(12, 6))
sns.lineplot(x='Month', y='Sales', hue='Year', data=monthly_sales)
plt.title('Monthly Sales Trends')
plt.xlabel('Month')
plt.ylabel('Sales')
plt.show()

# Analyze Product Category Preferences
category_sales = data.groupby('Category')['Sales'].sum().sort_values(ascending=False)
plt.figure(figsize=(10, 5))
sns.barplot(x=category_sales.index, y=category_sales.values)
plt.title('Top-Selling Product Categories')
plt.xlabel('Category')
plt.ylabel('Sales')
plt.xticks(rotation=45)
plt.show()

# Cluster Customers Based on Spending (Example with KMeans)
from sklearn.cluster import KMeans

# Prepare data for clustering
customer_data = data.groupby('CustomerID')['Sales'].sum().reset_index()
kmeans = KMeans(n_clusters=3)
customer_data['Cluster'] = kmeans.fit_predict(customer_data[['Sales']])

# Visualize clusters
plt.figure(figsize=(8, 5))
sns.scatterplot(x='CustomerID', y='Sales', hue='Cluster', data=customer_data, palette='viridis')
plt.title('Customer Segments')
plt.xlabel('Customer ID')
plt.ylabel('Total Spending')
plt.show()

# Save cleaned data
data.to_csv('cleaned_shopping_data.csv', index=False)
print("Data processing complete. Cleaned data saved as 'cleaned_shopping_data.csv'.") 
