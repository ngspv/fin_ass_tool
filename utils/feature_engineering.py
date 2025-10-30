import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.feature_selection import SelectKBest, f_classif
import logging
import warnings
warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FinancialFeatureEngineer:

    def __init__(self):
        self.scalers = {}
        self.encoders = {}
        self.feature_selector = None
        self.feature_names = []
        self.all_feature_names = []
        self.is_fitted = False
        self.amount_quantiles = None
        self.amount_bins = None
        self.amount_labels = None
        self.amount_mean = None
        self.amount_std = None
        self.start_date = None

    def create_temporal_features(self, df):
        df = df.copy()
        df['transaction_date'] = pd.to_datetime(df['transaction_date'])
        if not hasattr(self, 'start_date') or self.start_date is None:
            self.start_date = df['transaction_date'].min()
        df['days_since_start'] = (
            df['transaction_date'] - self.start_date).dt.days
        df['year'] = df['transaction_date'].dt.year
        df['month'] = df['transaction_date'].dt.month
        df['day'] = df['transaction_date'].dt.day
        df['day_of_week'] = df['transaction_date'].dt.dayofweek
        df['day_of_year'] = df['transaction_date'].dt.dayofyear
        df['quarter'] = df['transaction_date'].dt.quarter
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        df['hour_sin'] = np.sin(2 * np.pi * df['hour_of_day'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour_of_day'] / 24)
        return df

    def create_amount_features(self, df):
        df = df.copy()
        df['log_amount'] = np.log10(df['amount'] + 1)
        df['sqrt_amount'] = np.sqrt(df['amount'])
        if not hasattr(self, 'amount_mean') or self.amount_mean is None:
            self.amount_mean = df['amount'].mean()
            self.amount_std = df['amount'].std()
        df['amount_zscore'] = (
            df['amount'] - self.amount_mean) / self.amount_std
        df['amount_percentile'] = df['amount'].rank(pct=True)
        if self.amount_quantiles is None:
            amount_quantiles = df['amount'].quantile([0.25, 0.5, 0.75, 0.9])
            raw_bins = [0, amount_quantiles[0.25], amount_quantiles[0.5],
                        amount_quantiles[0.75], amount_quantiles[0.9], np.inf]
            unique_bins = []
            for i, bin_val in enumerate(raw_bins):
                if i == 0 or bin_val != unique_bins[-1]:
                    unique_bins.append(bin_val)
            if len(unique_bins) < 2:
                amount_min = df['amount'].min()
                amount_max = df['amount'].max()
                if amount_min == amount_max:
                    unique_bins = [amount_min - 0.1, amount_max + 0.1]
                    labels = ['Medium']
                else:
                    unique_bins = [
                        amount_min - 0.1, (amount_min + amount_max) / 2, amount_max + 0.1]
                    labels = ['Low', 'High']
            else:
                n_bins = len(unique_bins) - 1
                if n_bins >= 5:
                    labels = ['Very_Low', 'Low', 'Medium',
                              'High', 'Very_High'][:n_bins]
                elif n_bins == 4:
                    labels = ['Low', 'Medium', 'High', 'Very_High']
                elif n_bins == 3:
                    labels = ['Low', 'Medium', 'High']
                elif n_bins == 2:
                    labels = ['Low', 'High']
                else:
                    labels = ['Medium']
            self.amount_quantiles = amount_quantiles
            self.amount_bins = unique_bins
            self.amount_labels = labels
        else:
            unique_bins = self.amount_bins
            labels = self.amount_labels
        df['amount_category'] = pd.cut(
            df['amount'], bins=unique_bins, labels=labels, duplicates='drop', include_lowest=True)
        return df

    def create_frequency_features(self, df):
        df = df.copy()
        df['frequency_category'] = pd.cut(df['transaction_frequency'], bins=[
                                          0, 5, 10, 20, np.inf], labels=['Low', 'Medium', 'High', 'Very_High'])
        segment_avg_freq = df.groupby('user_segment')[
            'transaction_frequency'].transform('mean')
        df['frequency_vs_segment'] = df['transaction_frequency'] / segment_avg_freq
        return df

    def create_risk_features(self, df):
        df = df.copy()
        df['volatility_category'] = pd.cut(df['market_volatility'], bins=[
                                           0, 0.2, 0.4, 0.6, 1.0], labels=['Low', 'Medium', 'High', 'Very_High'])
        df['economic_sentiment'] = np.where(df['economic_indicator'] > 0.05, 'Positive', np.where(
            df['economic_indicator'] < -0.05, 'Negative', 'Neutral'))
        df['amount_volatility_interaction'] = df['log_amount'] * \
            df['market_volatility']
        df['frequency_volatility_interaction'] = df['transaction_frequency'] * \
            df['market_volatility']
        df['after_hours_risk'] = df['is_after_hours'].astype(
            int) * df['market_volatility']
        df['weekend_risk'] = df['is_weekend'].astype(
            int) * df['market_volatility']
        return df

    def create_aggregated_features(self, df):
        df = df.copy()
        segment_stats = df.groupby('user_segment').agg({'amount': ['mean', 'std', 'median'], 'market_volatility': [
            'mean', 'std'], 'transaction_frequency': ['mean', 'std']}).round(4)
        segment_stats.columns = ['_'.join(col).strip()
                                 for col in segment_stats.columns]
        segment_stats = segment_stats.add_prefix('segment_')
        asset_stats = df.groupby('asset_type').agg({'amount': ['mean', 'std'], 'market_volatility': [
            'mean', 'std'], 'risk_score': ['mean', 'std']}).round(4)
        asset_stats.columns = ['_'.join(col).strip()
                               for col in asset_stats.columns]
        asset_stats = asset_stats.add_prefix('asset_')
        segment_stats = segment_stats.fillna(0)
        asset_stats = asset_stats.fillna(0)
        df = df.merge(segment_stats, left_on='user_segment',
                      right_index=True, how='left')
        df = df.merge(asset_stats, left_on='asset_type',
                      right_index=True, how='left')
        df['amount_vs_segment_avg'] = df['amount'] / df['segment_amount_mean']
        df['amount_vs_asset_avg'] = df['amount'] / df['asset_amount_mean']
        df['volatility_vs_segment_avg'] = df['market_volatility'] / \
            df['segment_market_volatility_mean']
        df['volatility_vs_asset_avg'] = df['market_volatility'] / \
            df['asset_market_volatility_mean']
        return df

    def encode_categorical_features(self, df, categorical_columns=None):
        df = df.copy()
        if categorical_columns is None:
            categorical_columns = ['asset_type', 'user_segment', 'region', 'amount_category',
                                   'frequency_category', 'volatility_category', 'economic_sentiment']
        for col in categorical_columns:
            if col in df.columns:
                if col not in self.encoders:
                    self.encoders[col] = OneHotEncoder(
                        sparse_output=False, drop='first')
                    encoded_features = self.encoders[col].fit_transform(
                        df[[col]])
                else:
                    encoded_features = self.encoders[col].transform(df[[col]])
                feature_names = [
                    f'{col}_{cat}' for cat in self.encoders[col].categories_[0][1:]]
                encoded_df = pd.DataFrame(
                    encoded_features, columns=feature_names, index=df.index)
                df = pd.concat([df, encoded_df], axis=1)
                df = df.drop(columns=[col])
        return df

    def scale_numerical_features(self, df, numerical_columns=None):
        df = df.copy()
        if numerical_columns is None:
            numerical_columns = df.select_dtypes(
                include=[np.number]).columns.tolist()
            exclude_cols = ['risk_score',
                            'transaction_id', 'year', 'month', 'day']
            numerical_columns = [
                col for col in numerical_columns if col not in exclude_cols]
        for col in numerical_columns:
            if col in df.columns:
                if col not in self.scalers:
                    self.scalers[col] = StandardScaler()
                    df[col] = self.scalers[col].fit_transform(
                        df[[col]]).flatten()
                else:
                    df[col] = self.scalers[col].transform(df[[col]]).flatten()
        return df

    def select_features(self, df, target_column='risk_category', k=20):
        df = df.copy()
        X = df.drop(columns=[target_column, 'transaction_id',
                    'transaction_date'], errors='ignore')
        y = df[target_column]
        categorical_cols = X.select_dtypes(
            include=['object', 'category']).columns
        if len(categorical_cols) > 0:
            print(
                f'Warning: Found categorical columns during feature selection: {list(categorical_cols)}')
            print('Converting categorical columns to numeric using label encoding...')
            for col in categorical_cols:
                if col not in self.encoders:
                    self.encoders[f'{col}_label'] = LabelEncoder()
                    X[col] = self.encoders[f'{col}_label'].fit_transform(
                        X[col].astype(str))
                else:
                    X[col] = self.encoders[f'{col}_label'].transform(
                        X[col].astype(str))
        if y.dtype == 'object':
            if 'target_encoder' not in self.encoders:
                self.encoders['target_encoder'] = LabelEncoder()
                y_encoded = self.encoders['target_encoder'].fit_transform(y)
            else:
                y_encoded = self.encoders['target_encoder'].transform(y)
        else:
            y_encoded = y
        if self.feature_selector is None:
            self.feature_selector = SelectKBest(
                score_func=f_classif, k=min(k, X.shape[1]))
            X_selected = self.feature_selector.fit_transform(X, y_encoded)
            self.feature_names = X.columns[self.feature_selector.get_support(
            )].tolist()
        else:
            X_selected = self.feature_selector.transform(X)
        selected_df = pd.DataFrame(
            X_selected, columns=self.feature_names, index=df.index)
        selected_df[target_column] = y
        if 'transaction_id' in df.columns:
            selected_df['transaction_id'] = df['transaction_id']
        return selected_df

    def fit_transform(self, df, target_column='risk_category', select_features=True, k=20):
        logger.info(f'FIT_TRANSFORM: Starting with data shape {df.shape}')
        logger.info(f'FIT_TRANSFORM: Input columns: {list(df.columns)}')
        logger.info(f'FIT_TRANSFORM: Target column: {target_column}')
        logger.info(
            f'FIT_TRANSFORM: Select features: {select_features}, k={k}')
        print('Starting feature engineering pipeline...')
        print('Creating temporal features...')
        df = self.create_temporal_features(df)
        logger.info(
            f'FIT_TRANSFORM: After temporal features: {df.shape}, columns: {len(df.columns)}')
        print('Creating amount-based features...')
        df = self.create_amount_features(df)
        logger.info(
            f'FIT_TRANSFORM: After amount features: {df.shape}, columns: {len(df.columns)}')
        print('Creating frequency features...')
        df = self.create_frequency_features(df)
        logger.info(
            f'FIT_TRANSFORM: After frequency features: {df.shape}, columns: {len(df.columns)}')
        print('Creating risk features...')
        df = self.create_risk_features(df)
        logger.info(
            f'FIT_TRANSFORM: After risk features: {df.shape}, columns: {len(df.columns)}')
        print('Creating aggregated features...')
        df = self.create_aggregated_features(df)
        logger.info(
            f'FIT_TRANSFORM: After aggregated features: {df.shape}, columns: {len(df.columns)}')
        print('Encoding categorical features...')
        df = self.encode_categorical_features(df)
        logger.info(
            f'FIT_TRANSFORM: After encoding: {df.shape}, columns: {len(df.columns)}')
        print('Scaling numerical features...')
        df = self.scale_numerical_features(df)
        logger.info(
            f'FIT_TRANSFORM: After scaling: {df.shape}, columns: {len(df.columns)}')
        pre_selection_features = [col for col in df.columns if col not in [
            target_column, 'transaction_id', 'transaction_date']]
        logger.info(
            f'FIT_TRANSFORM: Features before selection ({len(pre_selection_features)}): {pre_selection_features}')
        self._training_feature_order = pre_selection_features.copy()
        logger.info(f'💾 FIT_TRANSFORM: Stored training feature order')
        if select_features:
            print(f'Selecting top {k} features...')
            df = self.select_features(df, target_column, k)
            logger.info(
                f'FIT_TRANSFORM: Feature selection completed. Selected features: {self.feature_names}')
        else:
            feature_cols = df.drop(columns=[
                                   target_column, 'transaction_id', 'transaction_date'], errors='ignore').columns.tolist()
            self.all_feature_names = feature_cols
            logger.info(
                f'FIT_TRANSFORM: No feature selection. All features stored: {self.all_feature_names}')
        self.is_fitted = True
        logger.info(f'🏁 FIT_TRANSFORM: Completed. Final shape: {df.shape}')
        print(f'Feature engineering completed. Final shape: {df.shape}')
        return df

    def transform(self, df, target_column='risk_category'):
        if not self.is_fitted:
            raise ValueError(
                'Feature engineer must be fitted before transform. Use fit_transform first.')
        logger.info(f'TRANSFORM: Starting with data shape {df.shape}')
        logger.info(f'TRANSFORM: Input columns: {list(df.columns)}')
        logger.info(f'TRANSFORM: Target column: {target_column}')
        print('Transforming new data...')
        df = self.create_temporal_features(df)
        logger.info(
            f'TRANSFORM: After temporal features: {df.shape}, columns: {len(df.columns)}')
        df = self.create_amount_features(df)
        logger.info(
            f'TRANSFORM: After amount features: {df.shape}, columns: {len(df.columns)}')
        df = self.create_frequency_features(df)
        logger.info(
            f'TRANSFORM: After frequency features: {df.shape}, columns: {len(df.columns)}')
        df = self.create_risk_features(df)
        logger.info(
            f'TRANSFORM: After risk features: {df.shape}, columns: {len(df.columns)}')
        df = self.create_aggregated_features(df)
        logger.info(
            f'TRANSFORM: After aggregated features: {df.shape}, columns: {len(df.columns)}')
        df = self.encode_categorical_features(df)
        logger.info(
            f'TRANSFORM: After encoding: {df.shape}, columns: {len(df.columns)}')
        df = self.scale_numerical_features(df)
        logger.info(
            f'TRANSFORM: After scaling: {df.shape}, columns: {len(df.columns)}')
        pre_selection_features = [col for col in df.columns if col not in [
            target_column, 'transaction_id', 'transaction_date']]
        logger.info(
            f'TRANSFORM: Features before selection/reordering ({len(pre_selection_features)}): {pre_selection_features}')
        if self.feature_selector is not None:
            logger.info(
                f'TRANSFORM: Applying feature selection. Expected features: {self.feature_names}')
            X = df.drop(
                columns=[target_column, 'transaction_id', 'transaction_date'], errors='ignore')
            logger.info(f'TRANSFORM: X shape before selection: {X.shape}')
            logger.info(
                f'TRANSFORM: X columns before selection: {list(X.columns)}')
            missing_features = set(self.feature_names) - set(X.columns)
            if missing_features:
                logger.error(
                    f' TRANSFORM: Missing features for selection: {missing_features}')
                raise ValueError(
                    f'Missing features for selection: {missing_features}')
            try:
                if hasattr(self, '_training_feature_order'):
                    logger.info(
                        f'TRANSFORM: Reordering features to match training order')
                    X_reordered = X[self._training_feature_order]
                    X_selected = self.feature_selector.transform(X_reordered)
                else:
                    logger.warning(
                        ' TRANSFORM: No training feature order stored, using current order')
                    X_selected = self.feature_selector.transform(X)
                logger.info(f'TRANSFORM: X_selected shape: {X_selected.shape}')
            except Exception as selection_error:
                logger.error(
                    f' TRANSFORM: Feature selection failed: {selection_error}')
                logger.info(f'TRANSFORM: Trying alternative approach...')
                try:
                    X_selected = X[self.feature_names].values
                    logger.info(
                        f'TRANSFORM: Manual feature selection successful: {X_selected.shape}')
                except Exception as manual_error:
                    logger.error(
                        f' TRANSFORM: Manual feature selection failed: {manual_error}')
                    raise manual_error
            selected_df = pd.DataFrame(
                X_selected, columns=self.feature_names, index=df.index)
            if target_column in df.columns:
                selected_df[target_column] = df[target_column]
            if 'transaction_id' in df.columns:
                selected_df['transaction_id'] = df['transaction_id']
            df = selected_df
            logger.info(
                f'TRANSFORM: Feature selection applied. Final features: {self.feature_names}')
        elif hasattr(self, 'all_feature_names') and self.all_feature_names:
            logger.info(
                f'TRANSFORM: Reordering features to match training. Expected features: {self.all_feature_names}')
            feature_cols = [
                col for col in self.all_feature_names if col in df.columns]
            other_cols = [col for col in df.columns if col not in feature_cols]
            df = df[feature_cols + other_cols]
            logger.info(
                f'TRANSFORM: Features reordered. Available features: {feature_cols}')
        else:
            logger.warning(
                ' TRANSFORM: No stored feature order found. Using current order.')
        final_features = [col for col in df.columns if col not in [
            target_column, 'transaction_id', 'transaction_date']]
        logger.info(f'🏁 TRANSFORM: Completed. Final shape: {df.shape}')
        logger.info(
            f'TRANSFORM: Final features ({len(final_features)}): {final_features}')
        print(f'Transformation completed. Final shape: {df.shape}')
        return df


def quick_feature_engineering(df, target_column='risk_category', k=15):
    logger.info('QUICK_FEATURE_ENGINEERING: Starting...')
    logger.info(f'QUICK_FE: Input data shape: {df.shape}')
    logger.info(f'QUICK_FE: Input columns: {list(df.columns)}')
    logger.info(f'QUICK_FE: Target column: {target_column}')
    logger.info(f'QUICK_FE: Number of features to select: {k}')
    if target_column in df.columns:
        target_dist = df[target_column].value_counts().to_dict()
        logger.info(f'QUICK_FE: Input target distribution: {target_dist}')
    else:
        logger.error(f" QUICK_FE: Target column '{target_column}' not found!")
        logger.error(f' QUICK_FE: Available columns: {list(df.columns)}')
    engineer = FinancialFeatureEngineer()
    logger.info('QUICK_FE: Created FinancialFeatureEngineer instance')
    logger.info('QUICK_FE: Calling fit_transform...')
    processed_df = engineer.fit_transform(
        df, target_column, select_features=True, k=k)
    logger.info('QUICK_FE: fit_transform completed')
    logger.info(f'QUICK_FE: Output data shape: {processed_df.shape}')
    logger.info(f'QUICK_FE: Output columns: {list(processed_df.columns)}')
    if target_column in processed_df.columns:
        output_target_dist = processed_df[target_column].value_counts(
        ).to_dict()
        logger.info(
            f'QUICK_FE: Output target distribution: {output_target_dist}')
        input_total = sum(target_dist.values()
                          ) if target_column in df.columns else 0
        output_total = sum(output_target_dist.values())
        if input_total != output_total:
            logger.error(
                f' QUICK_FE: DATA LOSS DETECTED! Input total: {input_total}, Output total: {output_total}')
        else:
            logger.info('QUICK_FE: No data loss detected')
    return (processed_df, engineer)


if __name__ == '__main__':
    from data_generator import generate_transaction_data
    print('Generating sample data...')
    df = generate_transaction_data(500)
    processed_df, engineer = quick_feature_engineering(df)
    print(f'\nOriginal data shape: {df.shape}')
    print(f'Processed data shape: {processed_df.shape}')
    print(f'\nSelected features: {engineer.feature_names}')
    print(f'\nProcessed data preview:')
    print(processed_df.head())
