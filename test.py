import pandas as pd

# Data Source 1: Website customers
website_data = pd.DataFrame({
    'customer_id': [1, 2, 3, 4],
    'full_name': ['John Doe', 'Jane Smith', 'Bob Johnson', 'Alice Brown'],
    'email_address': ['john@email.com', 'jane@email.com', 'bob@email.com', 'alice@email.com'],
    'purchase_total': [150.50, 89.99, 234.75, 67.80]
})

# Data Source 2: Mobile app users
mobile_data = pd.DataFrame({
    'user_id': [5, 6, 7, 8],
    'name': ['Charlie Wilson', 'Diana Prince', 'Eve Adams', 'Frank Miller'],
    'email': ['charlie@email.com', 'diana@email.com', 'eve@email.com', 'frank@email.com'],
    'app_spending': [45.99, 156.30, 78.45, 123.67],
    'device': ['iOS', 'Android', 'iOS', 'Android']
})

# Data Source 3: Social media leads
social_data = pd.DataFrame({
    'lead_id': [9, 10, 11, 12],
    'username': ['grace_h', 'henry_k', 'ivy_j', 'jack_l'],
    'contact_email': ['grace@email.com', 'henry@email.com', 'ivy@email.com', 'jack@email.com'],
    'followers': [1250, 890, 2340, 567],
    'engagement': [3.5, 4.2, 2.8, 5.1]
})

# Method 1: Simple concatenation - keeps all columns
def combine_different_datasets(datasets, source_names):
    combined_data = []
    print(enumerate(datasets))
    for i, df in enumerate(datasets):
        # Add source identifier
        df_copy = df.copy()
        df_copy['data_source'] = source_names[i]
        combined_data.append(df_copy)
    
    # Combine all datasets
    result = pd.concat(combined_data, ignore_index=True, sort=False)
    return result

# Combine the datasets
datasets = [website_data, mobile_data, social_data]
source_names = ['website', 'mobile_app', 'social_media']

combined_df = combine_different_datasets(datasets, source_names)
print("Combined DataFrame:")
print(combined_df)
print(f"\nShape: {combined_df.shape}")
print(f"Columns: {combined_df.columns.tolist()}")


# Method 2: Standardize column names before combining
def standardize_and_combine(datasets, column_mappings, source_names):
    standardized_data = []
    
    for i, df in enumerate(datasets):
        df_std = df.copy()
        
        # Apply column mappings for this dataset
        if source_names[i] in column_mappings:
            df_std = df_std.rename(columns=column_mappings[source_names[i]])
        
        # Add source information
        df_std['source'] = source_names[i]
        standardized_data.append(df_std)
    
    # Combine standardized datasets
    result = pd.concat(standardized_data, ignore_index=True, sort=False)
    return result

# Define column mappings
column_mappings = {
    'website': {
        'customer_id': 'id',
        'full_name': 'name',
        'email_address': 'email',
        'purchase_total': 'amount'
    },
    'mobile_app': {
        'user_id': 'id',
        'name': 'name',
        'email': 'email',
        'app_spending': 'amount'
    },
    'social_media': {
        'lead_id': 'id',
        'username': 'name',
        'contact_email': 'email'
    }
}

standardized_df = standardize_and_combine(datasets, column_mappings, source_names)
print("Standardized DataFrame:")
print(standardized_df)

# Method 3: Extract only specific columns from each dataset
def extract_common_columns(datasets, column_selections, source_names):
    extracted_data = []
    
    for i, df in enumerate(datasets):
        # Select only specified columns
        if source_names[i] in column_selections:
            selected_cols = column_selections[source_names[i]]
            df_selected = df[selected_cols].copy()
            df_selected['source'] = source_names[i]
            extracted_data.append(df_selected)
    
    # Combine extracted data
    result = pd.concat(extracted_data, ignore_index=True, sort=False)
    return result

# Define which columns to extract from each source
column_selections = {
    'website': ['customer_id', 'full_name', 'email_address'],
    'mobile_app': ['user_id', 'name', 'email'],
    'social_media': ['lead_id', 'username', 'contact_email']
}

extracted_df = extract_common_columns(datasets, column_selections, source_names)
print("Extracted DataFrame:")
print(extracted_df)


# Real-world example
def create_customer_master_data():
    # Different data sources with different structures
    crm_data = pd.DataFrame({
        'crm_id': [101, 102, 103],
        'customer_name': ['Alice Johnson', 'Bob Smith', 'Carol Davis'],
        'email': ['alice@example.com', 'bob@example.com', 'carol@example.com'],
        'phone': ['555-0101', '555-0102', '555-0103'],
        'registration_date': ['2023-01-15', '2023-02-20', '2023-03-10']
    })
    
    support_data = pd.DataFrame({
        'ticket_customer_id': [201, 202, 203],
        'full_name': ['Alice Johnson', 'David Wilson', 'Eva Brown'],
        'contact_email': ['alice@example.com', 'david@example.com', 'eva@example.com'],
        'support_tier': ['Premium', 'Standard', 'Premium'],
        'last_contact': ['2023-05-01', '2023-04-15', '2023-05-10']
    })
    
    # Combine using the integration function
    customer_sources = {
        'CRM_System': crm_data,
        'Support_System': support_data
    }
    
    master_customer_data = integrate_heterogeneous_data(customer_sources)
    return master_customer_data

# Create master customer data
master_data = create_customer_master_data()
print("Master Customer Data:")
print(master_data)