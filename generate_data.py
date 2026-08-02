"""
generate_data.py
Generates a synthetic banking transaction dataset with:
  - Normal customer-to-customer transactions (organic social/family network)
  - Injected "Smurfing" rings (1 kingpin -> many mule accounts -> 1 offshore sink),
    each transaction kept under the $10,000 CTR reporting threshold.

Output: data/transactions.csv, data/accounts.csv (with ground-truth labels used ONLY for evaluation)
"""
import pandas as pd
import numpy as np
from faker import Faker
import random
import uuid
from datetime import datetime, timedelta

random.seed(42)
np.random.seed(42)
fake = Faker()
Faker.seed(42)

N_NORMAL_ACCOUNTS = 2000
N_NORMAL_TXNS = 50000
N_RINGS = 8                 # number of separate smurfing rings
MULES_PER_RING = (15, 60)   # random range of mule accounts per ring
START_DATE = datetime(2025, 1, 1)


def make_accounts():
    accounts = []
    for i in range(N_NORMAL_ACCOUNTS):
        accounts.append({
            "account_id": f"ACC_{i:05d}",
            "name": fake.name(),
            "branch_code": random.choice(["BR-DEL", "BR-MUM", "BR-BLR", "BR-LKO", "BR-KOL"]),
            "account_type": random.choice(["SAVINGS", "CURRENT"]),
            "is_smurf_related": False,
            "role": "normal",
        })
    return pd.DataFrame(accounts)


def random_timestamp(day_span=180):
    return START_DATE + timedelta(
        days=random.randint(0, day_span),
        hours=random.randint(0, 23),
        minutes=random.randint(0, 59),
    )


def generate_normal_transactions(accounts_df):
    ids = accounts_df["account_id"].tolist()
    rows = []
    # give each account a small set of "regular contacts" -> organic clustering
    contacts = {acc: random.sample(ids, k=random.randint(2, 6)) for acc in ids}
    for _ in range(N_NORMAL_TXNS):
        sender = random.choice(ids)
        receiver = random.choice(contacts[sender]) if random.random() < 0.85 else random.choice(ids)
        if sender == receiver:
            continue
        amount = round(np.random.gamma(shape=2.0, scale=250) + 5, 2)
        rows.append({
            "transaction_id": str(uuid.uuid4())[:8],
            "timestamp": random_timestamp(),
            "sender_account": sender,
            "receiver_account": receiver,
            "amount": amount,
            "device_id": f"DEV_{random.randint(1, 900)}",
        })
    return pd.DataFrame(rows)


def generate_smurfing_ring(ring_idx, extra_accounts):
    """Kingpin -> N mules -> offshore sink, all within a tight time window, amounts just under $10k reporting limit."""
    n_mules = random.randint(*MULES_PER_RING)
    kingpin = f"KING_{ring_idx:02d}"
    sink = f"SINK_{ring_idx:02d}"
    mules = [f"MULE_{ring_idx:02d}_{j:03d}" for j in range(n_mules)]

    new_accounts = [
        {"account_id": kingpin, "name": fake.name(), "branch_code": "BR-DEL",
         "account_type": "CURRENT", "is_smurf_related": True, "role": "kingpin"},
        {"account_id": sink, "name": fake.name(), "branch_code": "BR-MUM",
         "account_type": "CURRENT", "is_smurf_related": True, "role": "sink"},
    ]
    for m in mules:
        new_accounts.append({"account_id": m, "name": fake.name(), "branch_code": "BR-BLR",
                              "account_type": "SAVINGS", "is_smurf_related": True, "role": "mule"})
    extra_accounts.extend(new_accounts)

    rows = []
    burst_start = random_timestamp(day_span=150)
    for j, m in enumerate(mules):
        # Kingpin splits money into sub-$10k chunks to each mule, in a tight burst
        amt = round(random.uniform(8500, 9900), 2)
        ts = burst_start + timedelta(minutes=random.randint(0, 90))
        rows.append({
            "transaction_id": str(uuid.uuid4())[:8],
            "timestamp": ts,
            "sender_account": kingpin,
            "receiver_account": m,
            "amount": amt,
            "device_id": "DEV_777",  # same device -> another red flag
        })
        # Mule immediately forwards to the offshore sink, minus a small "fee"
        fwd_ts = ts + timedelta(minutes=random.randint(5, 45))
        rows.append({
            "transaction_id": str(uuid.uuid4())[:8],
            "timestamp": fwd_ts,
            "sender_account": m,
            "receiver_account": sink,
            "amount": round(amt * random.uniform(0.95, 0.99), 2),
            "device_id": f"DEV_{random.randint(1, 900)}",
        })
    return pd.DataFrame(rows)


def main():
    normal_accounts = make_accounts()
    normal_txns = generate_normal_transactions(normal_accounts)

    extra_accounts = []
    ring_txns = [generate_smurfing_ring(i, extra_accounts) for i in range(N_RINGS)]

    all_accounts = pd.concat([normal_accounts, pd.DataFrame(extra_accounts)], ignore_index=True)
    all_txns = pd.concat([normal_txns] + ring_txns, ignore_index=True).sort_values("timestamp").reset_index(drop=True)

    all_accounts.to_csv("data/accounts.csv", index=False)
    all_txns.to_csv("data/transactions.csv", index=False)

    print(f"Accounts: {len(all_accounts)} ({all_accounts['is_smurf_related'].sum()} smurf-related)")
    print(f"Transactions: {len(all_txns)}")


if __name__ == "__main__":
    main()
