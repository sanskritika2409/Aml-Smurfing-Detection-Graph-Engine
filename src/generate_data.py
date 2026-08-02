"""
generate_data.py
Generates synthetic customer profile, clickstream, and transaction data for the
Customer Digital Twin project, with realistic churn-correlated behavioral decay
patterns baked in (so the downstream churn model has real signal to learn from).
"""
import pandas as pd
import numpy as np
from faker import Faker
import random
from datetime import datetime, timedelta

random.seed(7)
np.random.seed(7)
fake = Faker()
Faker.seed(7)

N_USERS = 6000
OBSERVATION_DAYS = 180
PRODUCT_CATEGORIES = ["Electronics", "Fashion", "Home", "Beauty", "Sports", "Books", "Grocery", "Toys"]
END_DATE = datetime(2025, 12, 31)
START_DATE = END_DATE - timedelta(days=OBSERVATION_DAYS)


def make_users():
    rows = []
    for i in range(N_USERS):
        signup_date = fake.date_between(start_date="-3y", end_date="-30d")
        plan = np.random.choice(["Free", "Pro", "Enterprise"], p=[0.55, 0.35, 0.10])
        # underlying "true" engagement propensity, drives both behavior and churn
        base_engagement = np.clip(np.random.beta(2, 2), 0.02, 0.98)
        # each user has a latent preference for 1-2 product categories (drives realistic purchase patterns
        # so the recommender has genuine signal to learn, rather than fully random category choice)
        n_pref = random.choice([1, 1, 2])
        preferred_categories = random.sample(PRODUCT_CATEGORIES, k=n_pref)
        rows.append({
            "user_id": f"U_{i:05d}",
            "signup_date": signup_date,
            "plan_type": plan,
            "region": random.choice(["North", "South", "East", "West"]),
            "_base_engagement": base_engagement,  # latent, not a real feature - used only to simulate behavior
            "_preferred_categories": preferred_categories,  # latent, not a real feature
        })
    return pd.DataFrame(rows)


def simulate_user_activity(users: pd.DataFrame):
    sessions, orders = [], []

    for _, u in users.iterrows():
        uid = u["user_id"]
        engagement = u["_base_engagement"]

        # decide if this user is a "churner" during the observation window
        will_churn = np.random.random() > engagement  # low engagement -> more likely to churn
        decay_start = np.random.randint(int(OBSERVATION_DAYS * 0.3), int(OBSERVATION_DAYS * 0.8)) if will_churn else None

        n_sessions_base = max(1, int(np.random.poisson(lam=engagement * 40) + 1))
        session_days = sorted(np.random.choice(range(OBSERVATION_DAYS), size=min(n_sessions_base, OBSERVATION_DAYS), replace=False))

        for day_offset in session_days:
            # if churner and past decay point, sharply reduce probability of session occurring
            if will_churn and decay_start is not None and day_offset > decay_start:
                if np.random.random() > 0.08:  # 92% chance the session is "skipped" post decay
                    continue
            ts = START_DATE + timedelta(days=int(day_offset), hours=random.randint(6, 23))
            duration = max(0.3, np.random.normal(loc=engagement * 12, scale=3))
            if will_churn and decay_start is not None and day_offset > decay_start:
                duration *= 0.3  # shorter, disengaged sessions near the end
            pages = max(1, int(duration * random.uniform(1.5, 3)))
            sessions.append({
                "user_id": uid,
                "session_date": ts,
                "duration_minutes": round(duration, 2),
                "pages_viewed": pages,
                "device": random.choice(["mobile", "desktop", "tablet"]),
            })

        # purchases correlate with engagement and plan
        plan_multiplier = {"Free": 0.5, "Pro": 1.2, "Enterprise": 2.0}[u["plan_type"]]
        n_orders = max(0, int(np.random.poisson(lam=engagement * 6 * plan_multiplier)))
        order_days = sorted(np.random.choice(range(OBSERVATION_DAYS), size=min(n_orders, OBSERVATION_DAYS), replace=False)) if n_orders else []
        preferred_categories = u["_preferred_categories"]
        for day_offset in order_days:
            if will_churn and decay_start is not None and day_offset > decay_start:
                if np.random.random() > 0.15:
                    continue
            ts = START_DATE + timedelta(days=int(day_offset))
            # 75% chance the purchase matches the user's latent preferred categor(y/ies), 25% exploration
            category = random.choice(preferred_categories) if np.random.random() < 0.75 else random.choice(PRODUCT_CATEGORIES)
            revenue = round(np.random.gamma(shape=2.0, scale=25) * plan_multiplier + 5, 2)
            orders.append({
                "user_id": uid,
                "order_date": ts,
                "product_category": category,
                "revenue": revenue,
            })

    return pd.DataFrame(sessions), pd.DataFrame(orders), users.assign(will_churn_label=[
        np.random.random() > e for e in users["_base_engagement"]
    ])


def main():
    users = make_users()
    sessions, orders, users_labeled = simulate_user_activity(users)

    # ground-truth churn label: no session AND no order in the final 30 days of the window
    cutoff = END_DATE - timedelta(days=30)
    active_recent = set(sessions[sessions["session_date"] >= cutoff]["user_id"]) | set(
        orders[orders["order_date"] >= cutoff]["user_id"]
    )
    users_labeled["churn_label"] = (~users_labeled["user_id"].isin(active_recent)).astype(int)
    users_labeled = users_labeled.drop(columns=["_base_engagement", "_preferred_categories", "will_churn_label"])

    users_labeled.to_csv("data/users.csv", index=False)
    sessions.to_csv("data/sessions.csv", index=False)
    orders.to_csv("data/orders.csv", index=False)

    print(f"Users: {len(users_labeled)} | Churn rate: {users_labeled['churn_label'].mean():.1%}")
    print(f"Sessions: {len(sessions)} | Orders: {len(orders)}")


if __name__ == "__main__":
    main()
