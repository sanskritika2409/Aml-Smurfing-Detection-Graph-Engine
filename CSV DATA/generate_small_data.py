# generate_small_data.py – small CSV data for AML practice
import pandas as pd
import numpy as np
from faker import Faker
import random

fake = Faker()
np.random.seed(42)
random.seed(42)

# ---------- CUSTOMERS (100) ----------
customers = pd.DataFrame({
    'customer_id': [f'CUST_{i:04d}' for i in range(1, 101)],
    'name': [fake.name() for _ in range(100)],
    'nationality': [fake.country() for _ in range(100)],
    'occupation': np.random.choice(['salaried', 'business', 'student', 'unemployed'], 100),
    'risk_score': np.random.uniform(0, 100, 100).round(2)
})

# ---------- ACCOUNTS (150) ----------
accounts = pd.DataFrame({
    'account_id': [f'ACC_{i:04d}' for i in range(1, 151)],
    'customer_id': np.random.choice(customers['customer_id'], 150),
    'account_type': np.random.choice(['savings', 'checking', 'business'], 150, p=[0.5, 0.3, 0.2]),
    'open_date': [fake.date_between(start_date='-3y', end_date='-1y') for _ in range(150)]
})

# ---------- NORMAL TRANSACTIONS (5,000) ----------
start_date = pd.Timestamp('2023-01-01')
end_date = pd.Timestamp('2023-12-31')
normal_tx = []
for _ in range(5000):
    from_acc = np.random.choice(accounts['account_id'])
    to_acc = np.random.choice(accounts['account_id'])
    while to_acc == from_acc:
        to_acc = np.random.choice(accounts['account_id'])
    amount = round(np.random.lognormal(mean=5, sigma=1.2), 2)
    amount = min(amount, 20000)
    ts = fake.date_time_between(start_date=start_date, end_date=end_date)
    channel = np.random.choice(['online', 'branch', 'atm', 'wire'], p=[0.5, 0.2, 0.1, 0.2])
    normal_tx.append((from_acc, to_acc, amount, ts, channel, 'normal'))

normal_df = pd.DataFrame(normal_tx, columns=['from_account', 'to_account', 'amount', 'timestamp', 'channel', 'label'])

# ---------- SMURF RING (tiny) ----------
master = accounts.sample(1).iloc[0]['account_id']
mules = accounts.sample(5)['account_id'].tolist()
smurf_tx = []
for mule in mules:
    for _ in range(random.randint(5, 10)):
        amount = round(random.uniform(8000, 9900), 2)
        ts = fake.date_time_between(start_date=start_date, end_date=end_date)
        smurf_tx.append((mule, master, amount, ts, 'online', 'smurf'))
    # occasional layering back
    if random.random() > 0.7:
        amount_back = round(random.uniform(1000, 5000), 2)
        ts_back = fake.date_time_between(start_date=start_date, end_date=end_date)
        smurf_tx.append((master, mule, amount_back, ts_back, 'wire', 'smurf'))

smurf_df = pd.DataFrame(smurf_tx, columns=['from_account', 'to_account', 'amount', 'timestamp', 'channel', 'label'])

# ---------- COMBINE & SHUFFLE ----------
all_tx = pd.concat([normal_df, smurf_df], ignore_index=True)
all_tx = all_tx.sample(frac=1).reset_index(drop=True)   # unsorted!
all_tx.insert(0, 'transaction_id', [f'TX_{i:06d}' for i in range(1, len(all_tx)+1)])
all_tx['timestamp'] = pd.to_datetime(all_tx['timestamp'])

# ---------- SAVE CSVs ----------
customers.to_csv('customers.csv', index=False)
accounts.to_csv('accounts.csv', index=False)
all_tx.to_csv('transactions.csv', index=False)

print("✅ Small dataset created!")
print(f"Transactions: {len(all_tx)} (normal: {len(normal_df)}, smurf: {len(smurf_df)})")
print("CSV files: customers.csv, accounts.csv, transactions.csv")