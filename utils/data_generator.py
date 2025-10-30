import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random


def generate_transaction_data(n_transactions=1000, start_date='2020-01-01', end_date='2024-12-31'):
    np.random.seed(42)
    random.seed(42)
    start = datetime.strptime(start_date, '%Y-%m-%d')
    end = datetime.strptime(end_date, '%Y-%m-%d')
    date_range = (end - start).days
    asset_types = ['Stock', 'Bond', 'Forex', 'Commodity',
                   'Crypto', 'Real Estate', 'Derivatives']
    asset_risk_weights = [0.15, 0.05, 0.2, 0.12, 0.35, 0.08, 0.25]
    user_segments = ['Retail', 'Institutional', 'High_Net_Worth', 'Corporate']
    segment_risk_multipliers = [1.2, 0.8, 1.0, 0.9]
    data = []
    for i in range(n_transactions):
        transaction_id = f'TXN_{i + 1:06d}'
        random_days = random.randint(0, date_range)
        transaction_date = start + timedelta(days=random_days)
        asset_type = np.random.choice(asset_types, p=np.array(
            [0.25, 0.2, 0.15, 0.1, 0.1, 0.05, 0.15]))
        asset_risk_weight = asset_risk_weights[asset_types.index(asset_type)]
        user_segment = np.random.choice(
            user_segments, p=[0.4, 0.2, 0.25, 0.15])
        segment_multiplier = segment_risk_multipliers[user_segments.index(
            user_segment)]
        if asset_type == 'Real Estate':
            amount = np.random.lognormal(12, 1.5)
        elif asset_type == 'Derivatives':
            amount = np.random.lognormal(10, 2)
        else:
            amount = np.random.lognormal(8, 1.5)
        amount = round(amount, 2)
        base_frequency = np.random.poisson(5)
        if user_segment == 'Institutional':
            transaction_frequency = base_frequency * 3
        elif user_segment == 'Corporate':
            transaction_frequency = base_frequency * 2
        else:
            transaction_frequency = base_frequency
        market_volatility = np.random.beta(2, 5)
        economic_indicator = np.random.normal(0, 0.1)
        amount_risk = min(np.log10(amount) / 6, 1)
        frequency_risk = min(transaction_frequency / 20, 1)
        volatility_risk = market_volatility
        economic_risk = abs(economic_indicator)
        risk_percentile = np.random.random()
        if risk_percentile < 0.33:
            risk_category = 'Low'
            final_risk_score = np.random.uniform(0.0, 0.4)
        elif risk_percentile < 0.67:
            risk_category = 'Medium'
            final_risk_score = np.random.uniform(0.3, 0.7)
        else:
            risk_category = 'High'
            final_risk_score = np.random.uniform(0.6, 1.0)
        calculated_risk = (amount_risk * 0.3 + frequency_risk * 0.2 + asset_risk_weight *
                           0.3 + volatility_risk * 0.1 + economic_risk * 0.1) * segment_multiplier
        final_risk_score = 0.7 * calculated_risk + 0.3 * final_risk_score
        is_weekend = transaction_date.weekday() >= 5
        hour_of_day = np.random.randint(0, 24)
        is_after_hours = hour_of_day < 9 or hour_of_day > 17
        regions = ['North America', 'Europe',
                   'Asia Pacific', 'Latin America', 'Middle East']
        region = np.random.choice(regions, p=[0.3, 0.25, 0.25, 0.1, 0.1])
        data.append({'transaction_id': transaction_id, 'transaction_date': transaction_date.strftime('%Y-%m-%d'), 'amount': amount, 'asset_type': asset_type, 'user_segment': user_segment, 'transaction_frequency': transaction_frequency, 'market_volatility': round(
            market_volatility, 4), 'economic_indicator': round(economic_indicator, 4), 'region': region, 'is_weekend': is_weekend, 'hour_of_day': hour_of_day, 'is_after_hours': is_after_hours, 'risk_score': round(final_risk_score, 4), 'risk_category': risk_category})
    df = pd.DataFrame(data)
    df['log_amount'] = np.log10(df['amount'] + 1)
    df['amount_zscore'] = (
        df['amount'] - df['amount'].mean()) / df['amount'].std()
    df['days_since_start'] = (pd.to_datetime(
        df['transaction_date']) - pd.to_datetime(start_date)).dt.days
    return df


def save_sample_data(filepath='data/sample_transactions.csv', n_transactions=1000):
    df = generate_transaction_data(n_transactions)
    df.to_csv(filepath, index=False)
    print(
        f'Sample data with {n_transactions} transactions saved to {filepath}')
    return df


if __name__ == '__main__':
    sample_df = save_sample_data()
    print('\nDataset Overview:')
    print(f'Shape: {sample_df.shape}')
    print(f'\nRisk Category Distribution:')
    print(sample_df['risk_category'].value_counts())
    print(f'\nAsset Type Distribution:')
    print(sample_df['asset_type'].value_counts())
    print(f'\nBasic Statistics:')
    print(sample_df[['amount', 'risk_score',
          'transaction_frequency']].describe())
