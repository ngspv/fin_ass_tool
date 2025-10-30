import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.base import BaseEstimator, ClassifierMixin
import xgboost as xgb
import lightgbm as lgb
import joblib
import warnings
import logging
warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class XGBoostWithLabelMapping(BaseEstimator, ClassifierMixin):

    def __init__(self, **kwargs):
        self.xgb_params = kwargs
        self.xgb_model = None
        self.label_mapping = {}
        self.reverse_mapping = {}

    def fit(self, X, y):
        unique_labels = np.unique(y)
        self.label_mapping = {
            old_label: new_label for new_label, old_label in enumerate(unique_labels)}
        self.reverse_mapping = {
            new_label: old_label for old_label, new_label in self.label_mapping.items()}
        y_mapped = np.array([self.label_mapping[label] for label in y])
        self.xgb_model = xgb.XGBClassifier(**self.xgb_params)
        self.xgb_model.fit(X, y_mapped)
        return self

    def predict(self, X):
        if self.xgb_model is None:
            raise ValueError('Model must be fitted before making predictions')
        y_pred_mapped = self.xgb_model.predict(X)
        y_pred_original = np.array(
            [self.reverse_mapping[label] for label in y_pred_mapped])
        return y_pred_original

    def predict_proba(self, X):
        if self.xgb_model is None:
            raise ValueError('Model must be fitted before making predictions')
        return self.xgb_model.predict_proba(X)

    def get_params(self, deep=True):
        return self.xgb_params

    def set_params(self, **params):
        self.xgb_params.update(params)
        return self


class FinancialRiskPredictor:

    def __init__(self):
        self.models = {}
        self.label_encoder = LabelEncoder()
        self.best_model = None
        self.best_model_name = None
        self.feature_importance = None
        self.model_scores = {}

    def prepare_data(self, df, target_column, test_size=0.2, random_state=42):
        logger.info(f'PREPARE_DATA: Input data shape: {df.shape}')
        logger.info(f'PREPARE_DATA: Input columns: {list(df.columns)}')
        logger.info(f'PREPARE_DATA: Target column: {target_column}')
        if target_column in df.columns:
            target_dist = df[target_column].value_counts().to_dict()
            logger.info(
                f'PREPARE_DATA: Target distribution BEFORE processing: {target_dist}')
        else:
            logger.error(
                f"PREPARE_DATA: Target column '{target_column}' not found in dataframe!")
            logger.error(
                f'PREPARE_DATA: Available columns: {list(df.columns)}')
        X = df.drop(columns=[target_column, 'transaction_id'], errors='ignore')
        y = df[target_column]
        logger.info(f'PREPARE_DATA: Features shape after drop: {X.shape}')
        logger.info(f'PREPARE_DATA: Target shape: {y.shape}')
        logger.info(f'PREPARE_DATA: Target data type: {y.dtype}')
        logger.info(
            f'PREPARE_DATA: Target unique values: {sorted(y.unique())}')
        y_encoded = self.label_encoder.fit_transform(y)
        logger.info(f'PREPARE_DATA: Target encoded shape: {y_encoded.shape}')
        logger.info(
            f'PREPARE_DATA: Target encoded unique values: {sorted(np.unique(y_encoded))}')
        unique_classes, class_counts = np.unique(y_encoded, return_counts=True)
        min_class_count = np.min(class_counts)
        logger.info(
            f'PREPARE_DATA: Class counts: {dict(zip(unique_classes, class_counts))}')
        logger.info(f'PREPARE_DATA: Min class count: {min_class_count}')
        if min_class_count >= 2:
            stratify = y_encoded
            logger.info('PREPARE_DATA: Using stratified split')
        else:
            stratify = None
            logger.warning(
                f'PREPARE_DATA: Some classes have only {min_class_count} sample(s). Using random split instead of stratified split.')
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=test_size, random_state=random_state, stratify=stratify)
        logger.info(
            f'PREPARE_DATA: Train set - X shape: {X_train.shape}, y shape: {y_train.shape}')
        logger.info(
            f'PREPARE_DATA: Test set - X shape: {X_test.shape}, y shape: {y_test.shape}')
        unique_train_labels = np.unique(y_train)
        unique_test_labels = np.unique(y_test)
        all_unique_labels = np.unique(np.concatenate(
            [unique_train_labels, unique_test_labels]))
        logger.info(
            f'PREPARE_DATA: Unique train labels: {sorted(unique_train_labels)}')
        logger.info(
            f'PREPARE_DATA: Unique test labels: {sorted(unique_test_labels)}')
        logger.info(
            f'PREPARE_DATA: All unique labels: {sorted(all_unique_labels)}')
        label_mapping = {old_label: new_label for new_label,
                         old_label in enumerate(all_unique_labels)}
        logger.info(f'PREPARE_DATA: Label mapping: {label_mapping}')
        y_train_consecutive = np.array(
            [label_mapping[label] for label in y_train])
        y_test_consecutive = np.array(
            [label_mapping[label] for label in y_test])
        train_class_counts = pd.Series(
            y_train_consecutive).value_counts().to_dict()
        test_class_counts = pd.Series(
            y_test_consecutive).value_counts().to_dict()
        logger.info(
            f'PREPARE_DATA: FINAL train class counts: {train_class_counts}')
        logger.info(
            f'PREPARE_DATA: FINAL test class counts: {test_class_counts}')
        logger.info(
            f'PREPARE_DATA: FINAL min train class size: {min(train_class_counts.values())}')
        return (X_train, X_test, y_train_consecutive, y_test_consecutive)

    def get_adaptive_cv_folds(self, y_train, fast_mode=True):
        unique_classes, class_counts = np.unique(y_train, return_counts=True)
        min_class_count = np.min(class_counts)
        total_samples = len(y_train)
        if fast_mode and total_samples < 500:
            if min_class_count < 3:
                cv_folds = 2
            elif min_class_count < 10:
                cv_folds = 3
            else:
                cv_folds = 3
        elif min_class_count < 3:
            cv_folds = 2
        elif min_class_count < 5:
            cv_folds = 3
        elif min_class_count < 10:
            cv_folds = min(5, min_class_count)
        else:
            cv_folds = 5
        print(
            f'Using {cv_folds}-fold cross-validation (min class size: {min_class_count}, fast_mode: {fast_mode})')
        return cv_folds

    def train_logistic_regression(self, X_train, y_train, X_test, y_test, fast_mode=True):
        print('Training Logistic Regression...')
        start_time = time.time()
        cv_folds = self.get_adaptive_cv_folds(y_train, fast_mode)
        if fast_mode:
            combinations = 1 * cv_folds
            print(f'Fast mode: {combinations} total fits')
        else:
            combinations = 3 * 2 * 2 * cv_folds
            print(f'Full mode: {combinations} total fits')
        if fast_mode:
            param_grid = {'C': [1], 'max_iter': [1000], 'solver': ['lbfgs']}
        else:
            param_grid = {'C': [0.1, 1, 10], 'max_iter': [
                1000, 2000], 'solver': ['liblinear', 'lbfgs']}
        lr = LogisticRegression(random_state=42)
        grid_search = GridSearchCV(
            lr, param_grid, cv=cv_folds, scoring='f1_weighted', n_jobs=1)
        grid_search.fit(X_train, y_train)
        best_lr = grid_search.best_estimator_
        self.models['Logistic Regression'] = best_lr
        y_pred = best_lr.predict(X_test)
        metrics = self._calculate_metrics(
            y_test, y_pred, 'Logistic Regression')
        elapsed_time = time.time() - start_time
        print(f'Logistic Regression completed in {elapsed_time:.1f} seconds')
        return metrics

    def train_random_forest(self, X_train, y_train, X_test, y_test, fast_mode=True):
        print('Training Random Forest...')
        start_time = time.time()
        cv_folds = self.get_adaptive_cv_folds(y_train, fast_mode)
        if fast_mode:
            combinations = 1 * cv_folds
            print(f'Fast mode: {combinations} total fits')
        else:
            combinations = 2 * 3 * 2 * 2 * cv_folds
            print(f'Full mode: {combinations} total fits')
        if fast_mode:
            param_grid = {'n_estimators': [100], 'max_depth': [
                10], 'min_samples_split': [2], 'min_samples_leaf': [1]}
        else:
            param_grid = {'n_estimators': [100, 200], 'max_depth': [
                10, 20, None], 'min_samples_split': [2, 5], 'min_samples_leaf': [1, 2]}
        rf = RandomForestClassifier(random_state=42)
        grid_search = GridSearchCV(
            rf, param_grid, cv=cv_folds, scoring='f1_weighted', n_jobs=1)
        grid_search.fit(X_train, y_train)
        best_rf = grid_search.best_estimator_
        self.models['Random Forest'] = best_rf
        y_pred = best_rf.predict(X_test)
        metrics = self._calculate_metrics(y_test, y_pred, 'Random Forest')
        elapsed_time = time.time() - start_time
        print(f'Random Forest completed in {elapsed_time:.1f} seconds')
        return metrics

    def train_xgboost(self, X_train, y_train, X_test, y_test, fast_mode=True):
        print('Training XGBoost...')
        start_time = time.time()
        cv_folds = self.get_adaptive_cv_folds(y_train, fast_mode)
        if fast_mode:
            combinations = 1 * cv_folds
            print(f'Fast mode: {combinations} total fits')
        else:
            combinations = 2 * 3 * 2 * 2 * cv_folds
            print(f'Full mode: {combinations} total fits')
        if fast_mode:
            param_grid = {'n_estimators': [100], 'max_depth': [
                6], 'learning_rate': [0.1], 'subsample': [0.8]}
        else:
            param_grid = {'n_estimators': [100, 200], 'max_depth': [
                3, 6, 9], 'learning_rate': [0.1, 0.2], 'subsample': [0.8, 1.0]}
        xgb_model = XGBoostWithLabelMapping(
            random_state=42, eval_metric='mlogloss')
        grid_search = GridSearchCV(
            xgb_model, param_grid, cv=cv_folds, scoring='f1_weighted', n_jobs=1)
        grid_search.fit(X_train, y_train)
        best_xgb = grid_search.best_estimator_
        self.models['XGBoost'] = best_xgb
        y_pred = best_xgb.predict(X_test)
        metrics = self._calculate_metrics(y_test, y_pred, 'XGBoost')
        elapsed_time = time.time() - start_time
        print(f'XGBoost completed in {elapsed_time:.1f} seconds')
        return metrics

    def train_lightgbm(self, X_train, y_train, X_test, y_test, fast_mode=True):
        print('Training LightGBM...')
        cv_folds = self.get_adaptive_cv_folds(y_train, fast_mode)
        if fast_mode:
            param_grid = {'n_estimators': [100], 'max_depth': [
                6], 'learning_rate': [0.1], 'num_leaves': [31]}
        else:
            param_grid = {'n_estimators': [100, 200], 'max_depth': [
                3, 6, 9], 'learning_rate': [0.1, 0.2], 'num_leaves': [31, 50]}
        lgb_model = lgb.LGBMClassifier(
            random_state=42, verbose=-1, force_col_wise=True, boosting_type='gbdt')
        grid_search = GridSearchCV(
            lgb_model, param_grid, cv=cv_folds, scoring='f1_weighted', n_jobs=1)
        grid_search.fit(X_train, y_train)
        best_lgb = grid_search.best_estimator_
        self.models['LightGBM'] = best_lgb
        y_pred = best_lgb.predict(X_test)
        metrics = self._calculate_metrics(y_test, y_pred, 'LightGBM')
        return metrics

    def _calculate_metrics(self, y_test, y_pred, model_name):
        metrics = {'model': model_name, 'accuracy': accuracy_score(y_test, y_pred), 'precision': precision_score(
            y_test, y_pred, average='weighted'), 'recall': recall_score(y_test, y_pred, average='weighted'), 'f1_score': f1_score(y_test, y_pred, average='weighted')}
        self.model_scores[model_name] = metrics
        print(f"\n{model_name} Performance:")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"F1-Score: {metrics['f1_score']:.4f}")
        return metrics

    def cross_validate_models(self, X, y, cv_folds=None):
        if cv_folds is None:
            cv_folds = self.get_adaptive_cv_folds(y)
        print(f'\nPerforming {cv_folds}-fold cross-validation...')
        cv_results = {}
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        for model_name, model in self.models.items():
            cv_scores = cross_val_score(
                model, X, y, cv=skf, scoring='f1_weighted', n_jobs=1)
            cv_results[model_name] = {'mean_score': cv_scores.mean(
            ), 'std_score': cv_scores.std(), 'scores': cv_scores}
            print(
                f'{model_name}: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})')
        return cv_results

    def select_best_model(self):
        if not self.model_scores:
            raise ValueError('No models have been trained yet.')
        best_score = 0
        best_name = None
        for model_name, metrics in self.model_scores.items():
            if metrics['f1_score'] > best_score:
                best_score = metrics['f1_score']
                best_name = model_name
        if best_name is None and self.model_scores:
            best_name = list(self.model_scores.keys())[0]
            best_score = self.model_scores[best_name]['f1_score']
            print(
                f'Warning: All models have poor performance. Selecting {best_name} as default.')
        if best_name is None:
            raise ValueError('No trained models available for selection.')
        self.best_model_name = best_name
        self.best_model = self.models[best_name]
        print(f'\nBest model: {best_name} with F1-score: {best_score:.4f}')
        return (best_name, self.best_model)

    def get_feature_importance(self, feature_names):
        if self.best_model is None:
            self.select_best_model()
        if hasattr(self.best_model, 'feature_importances_'):
            importances = self.best_model.feature_importances_
        elif hasattr(self.best_model, 'coef_'):
            importances = np.abs(self.best_model.coef_[0])
        else:
            print('Feature importance not available for this model type.')
            return None
        feature_importance_df = pd.DataFrame(
            {'feature': feature_names, 'importance': importances}).sort_values('importance', ascending=False)
        self.feature_importance = feature_importance_df
        return feature_importance_df

    def predict_risk(self, X):
        if self.best_model is None:
            raise ValueError('No model has been selected. Train models first.')
        predictions = self.best_model.predict(X)
        probabilities = self.best_model.predict_proba(X)
        risk_categories = self.label_encoder.inverse_transform(predictions)
        return (risk_categories, probabilities)

    def predict_risk_score(self, X):
        if self.best_model is None:
            raise ValueError('No model has been selected. Train models first.')
        probabilities = self.best_model.predict_proba(X)
        class_names = self.label_encoder.classes_
        risk_weight_mapping = {'Low': 0.1, 'Medium': 0.5, 'High': 0.9}
        risk_weights = np.array([risk_weight_mapping[class_name]
                                for class_name in class_names])
        base_risk_scores = np.sum(probabilities * risk_weights, axis=1)
        scenario_adjustments = np.zeros(len(X))
        if 'market_volatility' in X.columns:
            volatility_impact = X['market_volatility'] * 0.3
            scenario_adjustments += volatility_impact
        if 'economic_indicator' in X.columns:
            econ_impact = np.abs(X['economic_indicator']) * 0.2
            econ_impact = np.where(
                X['economic_indicator'] < 0, econ_impact * 1.5, econ_impact)
            scenario_adjustments += econ_impact
        if 'transaction_frequency' in X.columns:
            freq_normalized = (X['transaction_frequency'] - X['transaction_frequency'].min()) / (
                X['transaction_frequency'].max() - X['transaction_frequency'].min() + 1e-08)
            freq_impact = freq_normalized * 0.15
            scenario_adjustments += freq_impact
        if 'amount' in X.columns:
            amount_normalized = (np.log1p(X['amount']) - np.log1p(X['amount']).min()) / (
                np.log1p(X['amount']).max() - np.log1p(X['amount']).min() + 1e-08)
            amount_impact = amount_normalized * 0.1
            scenario_adjustments += amount_impact
        enhanced_risk_scores = base_risk_scores + scenario_adjustments
        enhanced_risk_scores = np.clip(enhanced_risk_scores, 0.0, 1.0)
        scenario_seed = int(np.sum(enhanced_risk_scores * 1000) % 1000)
        np.random.seed(scenario_seed)
        noise = np.random.normal(0, 0.03, len(enhanced_risk_scores))
        final_scores = enhanced_risk_scores + noise
        final_scores = np.clip(final_scores, 0.05, 0.95)
        return final_scores

    def train_all_models(self, df, target_column='risk_category', test_size=0.2, fast_mode=True, include_lightgbm=True):
        logger.info(
            'TRAIN_ALL_MODELS: Starting comprehensive model training...')
        logger.info(f'TRAIN_ALL_MODELS: Input data shape: {df.shape}')
        logger.info(f'TRAIN_ALL_MODELS: Input columns: {list(df.columns)}')
        logger.info(f'TRAIN_ALL_MODELS: Target column: {target_column}')
        if target_column == 'risk_score' or (target_column in df.columns and df[target_column].dtype in ['float64', 'float32', 'int64', 'int32']):
            if target_column == 'risk_score':
                error_msg = "\nREGRESSION NOT SUPPORTED: The target 'risk_score' is a continuous numerical variable.\n\n📚 Current models are designed for CLASSIFICATION (predicting categories like Low/Medium/High).\nFor regression, you would need different models like:\n- Linear Regression\n- Random Forest Regressor  \n- XGBoost Regressor\n- Ridge/Lasso Regression\n\nRECOMMENDATION: Use 'risk_category' as target instead.\nThe trained classification models can already predict risk scores using predict_risk_score().\n                "
                print(error_msg)
                logger.error(error_msg.replace(
                    '\n', ' ').replace('  ', ' ').strip())
                raise ValueError(
                    "Regression targets like 'risk_score' are not supported. Use 'risk_category' instead.")
            else:
                error_msg = f"Target '{target_column}' appears to be numerical (dtype: {df[target_column].dtype}). Only categorical targets are supported."
                print(error_msg)
                logger.error(error_msg)
                raise ValueError(error_msg)
        if target_column in df.columns:
            target_dist = df[target_column].value_counts().to_dict()
            logger.info(
                f'TRAIN_ALL_MODELS: Input target distribution: {target_dist}')
        else:
            logger.error(
                f"TRAIN_ALL_MODELS: Target column '{target_column}' not found!")
        print('Starting comprehensive model training...')
        if fast_mode:
            print('Fast mode enabled - using reduced hyperparameter search')
            logger.info('TRAIN_ALL_MODELS: Fast mode enabled')
        if not include_lightgbm:
            print('Skipping LightGBM for faster training')
            logger.info('TRAIN_ALL_MODELS: Skipping LightGBM')
        logger.info('TRAIN_ALL_MODELS: Calling prepare_data...')
        X_train, X_test, y_train, y_test = self.prepare_data(
            df, target_column, test_size)
        logger.info('TRAIN_ALL_MODELS: prepare_data completed')
        logger.info('TRAIN_ALL_MODELS: Starting individual model training...')
        results = {}
        results['Logistic Regression'] = self.train_logistic_regression(
            X_train, y_train, X_test, y_test, fast_mode)
        logger.info('TRAIN_ALL_MODELS: Logistic Regression completed')
        results['Random Forest'] = self.train_random_forest(
            X_train, y_train, X_test, y_test, fast_mode)
        logger.info('TRAIN_ALL_MODELS: Random Forest completed')
        results['XGBoost'] = self.train_xgboost(
            X_train, y_train, X_test, y_test, fast_mode)
        logger.info('TRAIN_ALL_MODELS: XGBoost completed')
        if include_lightgbm:
            results['LightGBM'] = self.train_lightgbm(
                X_train, y_train, X_test, y_test, fast_mode)
            logger.info('TRAIN_ALL_MODELS: LightGBM completed')
        logger.info('TRAIN_ALL_MODELS: Preparing data for cross-validation...')
        X = df.drop(columns=[target_column, 'transaction_id'], errors='ignore')
        y = self.label_encoder.transform(df[target_column])
        logger.info(
            f'TRAIN_ALL_MODELS: CV data - X shape: {X.shape}, y shape: {y.shape}')
        logger.info(
            f'TRAIN_ALL_MODELS: CV target unique values: {sorted(np.unique(y))}')
        logger.info(
            f'TRAIN_ALL_MODELS: CV target distribution: {pd.Series(y).value_counts().to_dict()}')
        cv_results = self.cross_validate_models(X, y)
        self.select_best_model()
        feature_names = X.columns.tolist()
        self.get_feature_importance(feature_names)
        return {'test_results': results, 'cv_results': cv_results, 'best_model': self.best_model_name, 'feature_importance': self.feature_importance}


def quick_model_training(processed_df, target_column='risk_category', fast_mode=True, include_lightgbm=False):
    logger.info('QUICK_MODEL_TRAINING: Starting...')
    logger.info(
        f'QUICK_MODEL_TRAINING: Input data shape: {processed_df.shape}')
    logger.info(
        f'QUICK_MODEL_TRAINING: Input columns: {list(processed_df.columns)}')
    logger.info(f'QUICK_MODEL_TRAINING: Target column: {target_column}')
    logger.info(f'QUICK_MODEL_TRAINING: Fast mode: {fast_mode}')
    logger.info(f'QUICK_MODEL_TRAINING: Include LightGBM: {include_lightgbm}')
    string_columns = processed_df.select_dtypes(
        include=['object']).columns.tolist()
    if target_column in string_columns:
        string_columns.remove(target_column)
    problematic_cols = []
    for col in string_columns:
        if col != 'transaction_id':
            sample_values = processed_df[col].dropna().head(3).tolist()
            if any((str(val).count('-') >= 2 or str(val).count('/') >= 2 for val in sample_values)):
                problematic_cols.append(col)
            elif processed_df[col].nunique() > 10:
                problematic_cols.append(col)
    if problematic_cols:
        error_msg = f'RAW DATA DETECTED! String columns that need feature engineering: {problematic_cols}. Please apply feature engineering first using quick_feature_engineering() before model training.'
        logger.error(f'QUICK_MODEL_TRAINING: {error_msg}')
        raise ValueError(error_msg)
    logger.info('QUICK_MODEL_TRAINING: Data validation passed')
    if target_column in processed_df.columns:
        target_dist = processed_df[target_column].value_counts().to_dict()
        logger.info(
            f'QUICK_MODEL_TRAINING: Target distribution: {target_dist}')
        min_class_size = min(target_dist.values())
        if min_class_size <= 1:
            logger.error(
                f'QUICK_MODEL_TRAINING: CRITICAL - Min class size is {min_class_size}!')
            logger.error(
                'QUICK_MODEL_TRAINING: This will cause training failures!')
    else:
        logger.error(
            f"QUICK_MODEL_TRAINING: Target column '{target_column}' not found!")
        logger.error(
            f'QUICK_MODEL_TRAINING: Available columns: {list(processed_df.columns)}')
    predictor = FinancialRiskPredictor()
    logger.info('QUICK_MODEL_TRAINING: Created predictor instance')
    logger.info('QUICK_MODEL_TRAINING: Calling train_all_models...')
    results = predictor.train_all_models(
        processed_df, target_column, fast_mode=fast_mode, include_lightgbm=include_lightgbm)
    logger.info('QUICK_MODEL_TRAINING: train_all_models completed')
    return (predictor, results)


if __name__ == '__main__':
    import sys
    import os
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.append(parent_dir)
    from utils.data_generator import generate_transaction_data
    from utils.feature_engineering import quick_feature_engineering
    print('Generating sample data...')
    df = generate_transaction_data(500)
    print('Processing features...')
    processed_df, engineer = quick_feature_engineering(df, k=10)
    print('Training models...')
    predictor, results = quick_model_training(processed_df)
    print('\nTraining completed!')
    print(f'Best model: {predictor.best_model_name}')
    if predictor.feature_importance is not None:
        print('\nTop 5 features:')
        print(predictor.feature_importance.head(5))
