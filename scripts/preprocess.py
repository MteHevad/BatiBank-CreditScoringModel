import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Load the dataset
data = pd.read_csv('data.csv')

# Handle missing values: Fill missing ProductCategory with 'Unknown'
data['ProductCategory'].fillna('Unknown', inplace=True)

# Aggregate customer transaction data
data['TotalTransactionAmount'] = data.groupby('CustomerId')['Amount'].transform('sum')
data['TransactionCount'] = data.groupby('CustomerId')['TransactionId'].transform('count')

# Extracting time-based features from TransactionStartTime
data['TransactionStartTime'] = pd.to_datetime(data['TransactionStartTime'])
data['TransactionHour'] = data['TransactionStartTime'].dt.hour
data['TransactionDay'] = data['TransactionStartTime'].dt.day
data['TransactionMonth'] = data['TransactionStartTime'].dt.month

# Encoding categorical variables (e.g., ProductCategory and ChannelId)
encoder = OneHotEncoder()
encoded_data = pd.DataFrame(encoder.fit_transform(data[['ProductCategory', 'ChannelId']]).toarray(), 
                            columns=encoder.get_feature_names_out(['ProductCategory', 'ChannelId']))

# Normalize/Standardize numerical features (e.g., Amount)
scaler = StandardScaler()
data[['Amount', 'TotalTransactionAmount']] = scaler.fit_transform(data[['Amount', 'TotalTransactionAmount']])

# Save cleaned and engineered dataset
cleaned_data = pd.concat([data, encoded_data], axis=1)
cleaned_data.to_csv('cleaned_data.csv', index=False)

# Display first few rows of cleaned data
print(cleaned_data.head())
