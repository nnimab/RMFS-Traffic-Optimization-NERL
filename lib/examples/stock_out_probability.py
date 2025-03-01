import pandas as pd

# Read the data from the CSV file
df = pd.read_csv('raw_order.csv')

# Group by item_code and item_quantity and count unique order_id
grouped_df = df.groupby(['item_code', 'item_quantity'])['order_id'].nunique().reset_index()
grouped_df.columns = ['item_code', 'quantity', 'number_of_orders_that_use_that_quantity']

# Debug print: Check the grouped DataFrame
print("Grouped DataFrame:")
print(grouped_df)

# Calculate the total number of orders for each item_code
grouped_df['total_orders'] = grouped_df.groupby('item_code')['number_of_orders_that_use_that_quantity'].transform('sum')

# Debug print: Check the DataFrame with total orders
print("DataFrame with total orders:")
print(grouped_df)

# Calculate the demand probability
grouped_df['demand_probability'] = grouped_df['number_of_orders_that_use_that_quantity'] / grouped_df['total_orders']

# Drop the total_orders column as it is no longer needed
grouped_df.drop(columns='total_orders', inplace=True)

# Debug print: Check the final DataFrame
print("Final DataFrame with demand probability:")
print(grouped_df)

# Verify that the sum of demand probabilities for each item_code is 1
sum_demand_probability = grouped_df.groupby('item_code')['demand_probability'].sum()
print("Sum of demand probabilities for each item_code:")
print(sum_demand_probability)

# Save the resulting DataFrame to a CSV file
grouped_df.to_csv('result.csv', index=False)