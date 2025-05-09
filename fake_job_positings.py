import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Load data
job_postings = pd.read_csv('/Users/oliviajardine/Programming Project/Fake Job Posting/fake_job_postings.csv')

# Drop useless features
job_postings = job_postings.drop(columns='job_id')

#  Fill missing values in key categorical fields with 'Unknown'
fill_cols = ['location', 'department', 'employment_type', 'required_experience',
             'required_education', 'industry', 'function', 'description']
job_postings[fill_cols] = job_postings[fill_cols].fillna('Unknown')

# --- LOCATION HANDLING ---
# Split into city and country if possible
location_split = job_postings['location'].str.split(',', n=1, expand=True)
job_postings['city'] = location_split[0].str.strip()
job_postings['country'] = location_split[1].str.strip().fillna('Unknown')

# --- TEXT LENGTH FEATURE ---
job_postings['description_len'] = job_postings['description'].apply(len)

# --- DEFINE CATEGORICAL COLUMNS TO ENCODE ---
categorical_cols = ['city', 'country', 'department', 'employment_type',
                    'required_experience', 'required_education', 'industry', 'function']

# --- ONE-HOT ENCODE ALL CATEGORICAL FEATURES ---
categorical_dummies = pd.get_dummies(job_postings[categorical_cols], prefix=categorical_cols)
categorical_dummies = categorical_dummies.astype('int8')

# --- NUMERIC FEATURES ---
numeric_cols = ['telecommuting', 'has_company_logo', 'has_questions', 'description_len']

# --- FINAL FEATURE MATRIX ---
X = pd.concat([job_postings[numeric_cols], categorical_dummies], axis=1)

# --- TARGET VARIABLE ---
y = job_postings['fraudulent']

# --- OPTIONAL: Train/test split for model training ---
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

print(categorical_dummies.dtypes.head())
print(categorical_dummies.info())