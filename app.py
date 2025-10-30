from models.risk_predictor import FinancialRiskPredictor, quick_model_training
from utils.visualization import RiskVisualization, create_risk_summary_stats
from utils.feature_engineering import FinancialFeatureEngineer, quick_feature_engineering
from utils.data_generator import generate_transaction_data, save_sample_data
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'models'))
st.set_page_config(page_title='Financial Risk Assessment Tool',
                   layout='wide', initial_sidebar_state='expanded')
st.markdown('\n<style>\n    .main {\n        padding-top: 2rem;\n    }\n    .stAlert {\n        margin-top: 1rem;\n    }\n    .metric-container {\n        background-color:\n        padding: 1rem;\n        border-radius: 0.5rem;\n        margin: 0.5rem 0;\n    }\n</style>\n', unsafe_allow_html=True)
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False
if 'current_data' not in st.session_state:
    st.session_state.current_data = None
if 'predictor' not in st.session_state:
    st.session_state.predictor = None
if 'feature_engineer' not in st.session_state:
    st.session_state.feature_engineer = None


def main():
    st.title('AI-Powered Financial Risk Assessment Tool')
    st.markdown('\n    This application analyzes financial transaction data to predict risk levels using advanced machine learning models.\n    Upload your data or use our sample dataset to get started with risk assessment and visualization.\n    ')
    st.sidebar.title('Navigation')
    page = st.sidebar.selectbox('Select a page:', [
                                'Data Overview', 'Risk Prediction', 'Visualizations', 'Model Training', 'Scenario Analysis'])
    if page == 'Data Overview':
        data_overview_page()
    elif page == 'Risk Prediction':
        risk_prediction_page()
    elif page == 'Visualizations':
        visualizations_page()
    elif page == 'Model Training':
        model_training_page()
    elif page == 'Scenario Analysis':
        scenario_analysis_page()


def data_overview_page():
    st.header('Data Overview')
    st.subheader('1. Load Data')
    col1, col2 = st.columns(2)
    with col1:
        st.write('**Option A: Upload Your Data**')
        uploaded_file = st.file_uploader(
            'Choose a CSV file', type='csv', help='Upload a CSV file with financial transaction data')
        if uploaded_file is not None:
            try:
                data = pd.read_csv(uploaded_file)
                st.session_state.current_data = data
                st.session_state.data_loaded = True
                st.success(f'Data loaded successfully! Shape: {data.shape}')
            except Exception as e:
                st.error(f'Error loading data: {str(e)}')
    with col2:
        st.write('**Option B: Generate Sample Data**')
        n_transactions = st.number_input(
            'Number of transactions', min_value=100, max_value=5000, value=1000, step=100)
        if st.button('Generate Sample Data'):
            with st.spinner('Generating sample data...'):
                data = generate_transaction_data(n_transactions)
                st.session_state.current_data = data
                st.session_state.data_loaded = True
                st.success(f'Sample data generated! Shape: {data.shape}')
    if st.session_state.data_loaded and st.session_state.current_data is not None:
        st.subheader('2. Data Summary')
        data = st.session_state.current_data
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric('Total Transactions', len(data))
        with col2:
            high_risk_pct = len(
                data[data['risk_category'] == 'High']) / len(data) * 100
            st.metric('High Risk %', f'{high_risk_pct:.1f}%')
        with col3:
            avg_amount = data['amount'].mean()
            st.metric('Avg Amount', f'${avg_amount:,.2f}')
        with col4:
            avg_risk_score = data['risk_score'].mean()
            st.metric('Avg Risk Score', f'{avg_risk_score:.3f}')
        st.subheader('3. Data Preview')
        st.dataframe(data.head(10))
        st.subheader('4. Statistical Summary')
        st.dataframe(data.describe())
        if st.button('Download Current Data'):
            csv = data.to_csv(index=False)
            st.download_button(label='Download as CSV', data=csv,
                               file_name='financial_risk_data.csv', mime='text/csv')


def risk_prediction_page():
    st.header('Risk Prediction')
    if not st.session_state.data_loaded:
        st.warning('Please load data first from the Data Overview page.')
        return
    if not st.session_state.model_trained:
        st.warning('Please train a model first from the Model Training page.')
        return
    data = st.session_state.current_data
    predictor = st.session_state.predictor
    feature_engineer = st.session_state.feature_engineer
    st.subheader('Single Transaction Risk Assessment')
    with st.form('prediction_form'):
        col1, col2 = st.columns(2)
        with col1:
            amount = st.number_input(
                'Transaction Amount', min_value=0.01, value=1000.0, step=10.0)
            asset_type = st.selectbox('Asset Type', [
                                      'Stock', 'Bond', 'Forex', 'Commodity', 'Crypto', 'Real Estate', 'Derivatives'])
            user_segment = st.selectbox(
                'User Segment', ['Retail', 'Institutional', 'High_Net_Worth', 'Corporate'])
            region = st.selectbox('Region', [
                                  'North America', 'Europe', 'Asia Pacific', 'Latin America', 'Middle East'])
        with col2:
            transaction_frequency = st.number_input(
                'Monthly Transaction Frequency', min_value=1, max_value=100, value=5)
            market_volatility = st.slider(
                'Market Volatility', 0.0, 1.0, 0.3, 0.01)
            economic_indicator = st.slider(
                'Economic Indicator', -0.5, 0.5, 0.0, 0.01)
            is_weekend = st.checkbox('Weekend Transaction')
            is_after_hours = st.checkbox('After Hours Transaction')
            hour_of_day = st.slider('Hour of Day', 0, 23, 14)
        submitted = st.form_submit_button('Predict Risk')
        if submitted:
            try:
                new_transaction = pd.DataFrame({'transaction_id': ['NEW_001'], 'transaction_date': ['2024-01-15'], 'amount': [amount], 'asset_type': [asset_type], 'user_segment': [user_segment], 'region': [region], 'transaction_frequency': [
                                               transaction_frequency], 'market_volatility': [market_volatility], 'economic_indicator': [economic_indicator], 'is_weekend': [is_weekend], 'is_after_hours': [is_after_hours], 'hour_of_day': [hour_of_day], 'risk_score': [0.5], 'risk_category': ['Medium']})
                processed_transaction = feature_engineer.transform(
                    new_transaction)
                if hasattr(feature_engineer, 'feature_names') and feature_engineer.feature_names:
                    st.info(
                        f'Using stored feature names: {feature_engineer.feature_names}')
                    X_new = processed_transaction[feature_engineer.feature_names]
                elif hasattr(feature_engineer, 'all_feature_names') and feature_engineer.all_feature_names:
                    st.info(
                        f'Using stored all feature names: {feature_engineer.all_feature_names}')
                    X_new = processed_transaction[feature_engineer.all_feature_names]
                else:
                    st.warning(
                        'No stored feature order found - using dynamic extraction')
                    feature_columns = [col for col in processed_transaction.columns if col not in [
                        'risk_category', 'transaction_id']]
                    X_new = processed_transaction[feature_columns]
                    st.info(f'Dynamic features: {feature_columns}')
                st.info(f'Feature matrix shape: {X_new.shape}')
                st.info(f'Feature columns: {list(X_new.columns)}')
                risk_category, probabilities = predictor.predict_risk(X_new)
                risk_score = predictor.predict_risk_score(X_new)
                st.subheader('Prediction Results')
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric('Predicted Risk Category', risk_category[0])
                with col2:
                    st.metric('Risk Score', f'{risk_score[0]:.3f}')
                with col3:
                    confidence = np.max(probabilities[0]) * 100
                    st.metric('Confidence', f'{confidence:.1f}%')
                st.subheader('Risk Probability Breakdown')
                prob_df = pd.DataFrame(
                    {'Risk Level': predictor.label_encoder.classes_, 'Probability': probabilities[0]})
                fig_prob = px.bar(prob_df, x='Risk Level', y='Probability', title='Risk Level Probabilities',
                                  color='Risk Level', color_discrete_map={'Low': '#2E8B57', 'Medium': '#FF8C00', 'High': '#DC143C'})
                st.plotly_chart(fig_prob, use_container_width=True)
            except Exception as e:
                st.error(f'Error making prediction: {str(e)}')
    st.subheader('Batch Prediction')
    if st.button('Predict All Transactions'):
        with st.spinner('Making predictions for all transactions...'):
            try:
                processed_data = feature_engineer.transform(data)
                if hasattr(feature_engineer, 'feature_names') and feature_engineer.feature_names:
                    X_all = processed_data[feature_engineer.feature_names]
                elif hasattr(feature_engineer, 'all_feature_names') and feature_engineer.all_feature_names:
                    X_all = processed_data[feature_engineer.all_feature_names]
                else:
                    feature_columns = [col for col in processed_data.columns if col not in [
                        'risk_category', 'transaction_id']]
                    X_all = processed_data[feature_columns]
                predicted_categories, predicted_probs = predictor.predict_risk(
                    X_all)
                predicted_scores = predictor.predict_risk_score(X_all)
                data_with_predictions = data.copy()
                data_with_predictions['predicted_risk_category'] = predicted_categories
                data_with_predictions['predicted_risk_score'] = predicted_scores
                if 'risk_category' in data.columns:
                    accuracy = (data['risk_category'] ==
                                predicted_categories).mean()
                    st.success(
                        f'Predictions completed! Model accuracy: {accuracy:.2%}')
                st.dataframe(data_with_predictions[[
                             'transaction_id', 'amount', 'asset_type', 'risk_category', 'predicted_risk_category', 'predicted_risk_score']])
                csv = data_with_predictions.to_csv(index=False)
                st.download_button(label='Download Predictions', data=csv,
                                   file_name='risk_predictions.csv', mime='text/csv')
            except Exception as e:
                st.error(f'Error making batch predictions: {str(e)}')


def visualizations_page():
    st.header('Visualizations')
    if not st.session_state.data_loaded:
        st.warning('Please load data first from the Data Overview page.')
        return
    data = st.session_state.current_data
    viz = RiskVisualization()
    st.subheader('Risk Summary Statistics')
    stats = create_risk_summary_stats(data)
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric('Total Transactions', f"{stats['total_transactions']:,}")
    with col2:
        st.metric('High Risk Count', f"{stats['high_risk_count']:,}")
    with col3:
        st.metric('High Risk %', f"{stats['high_risk_percentage']:.1f}%")
    with col4:
        st.metric('Avg Risk Score', f"{stats['average_risk_score']:.3f}")
    st.subheader('Interactive Charts')
    chart_type = st.selectbox('Select visualization type:', [
                              'Risk Distribution', 'Amount vs Risk', 'Risk by Asset Type', 'Risk by User Segment', 'Risk Timeline', 'Volatility Impact'])
    try:
        if chart_type == 'Risk Distribution':
            fig = viz.plot_risk_distribution(data)
        elif chart_type == 'Amount vs Risk':
            fig = viz.plot_amount_vs_risk(data)
        elif chart_type == 'Risk by Asset Type':
            fig = viz.plot_risk_by_asset_type(data)
        elif chart_type == 'Risk by User Segment':
            fig = viz.plot_risk_by_user_segment(data)
        elif chart_type == 'Risk Timeline':
            fig = viz.plot_risk_timeline(data)
        elif chart_type == 'Volatility Impact':
            fig = viz.plot_volatility_impact(data)
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f'Error creating visualization: {str(e)}')
    if st.session_state.model_trained and st.session_state.predictor.feature_importance is not None:
        st.subheader('Feature Importance')
        fig_importance = viz.plot_feature_importance(
            st.session_state.predictor.feature_importance)
        st.plotly_chart(fig_importance, use_container_width=True)


def model_training_page():
    st.header('Model Training')
    if not st.session_state.data_loaded:
        st.warning('Please load data first from the Data Overview page.')
        return
    data = st.session_state.current_data
    st.subheader('Training Configuration')
    col1, col2 = st.columns(2)
    with col1:
        test_size = st.slider('Test Set Size', 0.1, 0.4, 0.2, 0.05)
        n_features = st.slider('Number of Features to Select', 10, 30, 15, 1)
        fast_mode = st.checkbox(
            'Fast Mode', value=True, help='Use reduced hyperparameter search for faster training')
    with col2:
        risk_columns = [col for col in data.columns if 'risk' in col.lower()]
        if 'risk_category' in risk_columns and 'risk_score' in risk_columns:
            risk_columns = ['risk_category'] + [col for col in risk_columns if col !=
                                                'risk_category' and col != 'risk_score']
        elif 'risk_category' in risk_columns:
            risk_columns = ['risk_category'] + \
                [col for col in risk_columns if col != 'risk_category']
        target_column = st.selectbox('Target Column', risk_columns, index=0)
        if target_column == 'risk_score':
            st.warning('**Risk Score Warning**: This is a numerical target requiring regression models. Current models are optimized for categorical prediction (risk_category). Training may be slow or fail.')
        elif target_column in data.columns and data[target_column].dtype in ['float64', 'float32', 'int64', 'int32']:
            st.warning(
                f"**Numerical Target Warning**: '{target_column}' appears to be numerical (dtype: {data[target_column].dtype}). Use categorical targets for best results.")
        include_lightgbm = st.checkbox(
            'Include LightGBM', value=False, help='LightGBM can be slow to train but may provide better results')
    if fast_mode:
        st.info('Fast mode: Reduced hyperparameter search for quicker training')
    if not include_lightgbm:
        st.info(
            'LightGBM disabled for faster training. Enable for potentially better accuracy.')
    if st.button('Start Training'):
        with st.spinner('Training models... This may take a few minutes.'):
            try:
                st.info('Validating input data...')
                string_columns = data.select_dtypes(
                    include=['object']).columns.tolist()
                if target_column in string_columns:
                    string_columns.remove(target_column)
                if len(string_columns) > 1:
                    date_cols = [
                        col for col in string_columns if 'date' in col.lower()]
                    if date_cols:
                        st.warning(
                            'Raw data detected with date columns. Feature engineering is required.')
                        st.info('String columns found: ' +
                                ', '.join(string_columns))
                st.info('Processing features...')
                processed_data, feature_engineer = quick_feature_engineering(
                    data, target_column, n_features)
                st.session_state.feature_engineer = feature_engineer
                processed_string_cols = processed_data.select_dtypes(
                    include=['object']).columns.tolist()
                if target_column in processed_string_cols:
                    processed_string_cols.remove(target_column)
                if len(processed_string_cols) > 1:
                    st.error(
                        'Feature engineering failed to convert all string columns!')
                    st.error(
                        f'Remaining string columns: {processed_string_cols}')
                    return
                st.success('Data validation passed - ready for model training')
                st.info('Training machine learning models...')
                predictor, results = quick_model_training(
                    processed_data, target_column, fast_mode=fast_mode, include_lightgbm=include_lightgbm)
                st.session_state.predictor = predictor
                st.session_state.model_trained = True
                st.success('Model training completed!')
                st.subheader('Training Results')
                model_comparison = pd.DataFrame(results['test_results']).T
                st.dataframe(model_comparison)
                col1, col2 = st.columns(2)
                with col1:
                    st.metric('Best Model', predictor.best_model_name)
                with col2:
                    best_score = predictor.model_scores[predictor.best_model_name]['f1_score']
                    st.metric('F1-Score', f'{best_score:.4f}')
                st.subheader('Cross-Validation Results')
                cv_df = pd.DataFrame(results['cv_results']).T
                st.dataframe(cv_df[['mean_score', 'std_score']])
                if results['feature_importance'] is not None:
                    st.subheader('Top Features')
                    st.dataframe(results['feature_importance'].head(10))
                st.subheader('Risk Distribution by Score')
                try:
                    processed_train_data = feature_engineer.transform(data)
                    if hasattr(feature_engineer, 'feature_names') and feature_engineer.feature_names:
                        X_train_viz = processed_train_data[feature_engineer.feature_names]
                    elif hasattr(feature_engineer, 'all_feature_names') and feature_engineer.all_feature_names:
                        X_train_viz = processed_train_data[feature_engineer.all_feature_names]
                    else:
                        feature_columns = [col for col in processed_train_data.columns if col not in [
                            'risk_category', 'transaction_id']]
                        X_train_viz = processed_train_data[feature_columns]
                    predicted_scores = predictor.predict_risk_score(
                        X_train_viz)
                    score_bins = np.linspace(0, 1, 11)
                    score_labels = [
                        f'{score_bins[i]:.1f}-{score_bins[i + 1]:.1f}' for i in range(len(score_bins) - 1)]
                    binned_scores = pd.cut(
                        predicted_scores, bins=score_bins, labels=score_labels, include_lowest=True)
                    score_dist = pd.DataFrame(
                        {'Score_Range': binned_scores, 'Actual_Risk': data[target_column] if target_column in data.columns else ['Unknown'] * len(data)})
                    dist_counts = score_dist.groupby(
                        ['Score_Range', 'Actual_Risk']).size().reset_index(name='Count')
                    fig_risk_dist = px.bar(dist_counts, x='Score_Range', y='Count', color='Actual_Risk', title='Risk Distribution by Predicted Score Range', labels={
                                           'Score_Range': 'Predicted Risk Score Range', 'Count': 'Number of Transactions'}, color_discrete_map={'Low': '#2E8B57', 'Medium': '#FF8C00', 'High': '#DC143C'})
                    fig_risk_dist.update_layout(xaxis_tickangle=45)
                    st.plotly_chart(fig_risk_dist, use_container_width=True)
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric('Min Risk Score',
                                  f'{np.min(predicted_scores):.3f}')
                    with col2:
                        st.metric('Avg Risk Score',
                                  f'{np.mean(predicted_scores):.3f}')
                    with col3:
                        st.metric('Max Risk Score',
                                  f'{np.max(predicted_scores):.3f}')
                except Exception as viz_error:
                    st.warning(
                        f'Could not generate risk distribution visualization: {str(viz_error)}')
            except ValueError as e:
                if 'RAW DATA DETECTED' in str(e):
                    st.error('**Data Preprocessing Required**')
                    st.markdown("\n                    **The system detected raw data that needs feature engineering:**\n                    \n                    - **What happened**: Your data contains string columns (like dates) that machine learning models cannot process directly\n                    - **What we're doing**: The app automatically applies feature engineering to convert these into numeric features  \n                    - **What you should know**: This is normal and expected - feature engineering is a critical step\n                    \n                    **Technical Details:**\n                    ")
                    st.code(str(e))
                    st.info(
                        "**Tip**: If you're using the API directly, always call `quick_feature_engineering()` before `quick_model_training()`")
                else:
                    st.error(f'**Validation Error**: {str(e)}')
                    st.info('Please check your data format and try again.')
            except Exception as e:
                st.error(f'**Unexpected Error During Training**')
                error_str = str(e).lower()
                if 'could not convert string to float' in error_str:
                    st.markdown("\n                    **Diagnosis**: String-to-numeric conversion error\n                    \n                    **Solution**: This usually means feature engineering didn't complete properly. \n                    - Try reducing the number of features\n                    - Check if your target column is correctly specified\n                    - Ensure your data has the expected format\n                    ")
                elif 'fit failed' in error_str or 'cross-validation' in error_str:
                    st.markdown('\n                    **Diagnosis**: Model training failed\n                    \n                    **Solutions**:\n                    - Try enabling Fast Mode for quicker training\n                    - Increase the minimum number of transactions (need at least 100)\n                    - Check if your target column has balanced classes\n                    ')
                st.exception(e)


def scenario_analysis_page():
    st.header('Scenario Analysis')
    if not st.session_state.model_trained:
        st.warning('Please train a model first from the Model Training page.')
        return
    st.subheader('Economic Scenario Simulation')
    col1, col2 = st.columns(2)
    with col1:
        st.write('**Market Conditions**')
        volatility_scenarios = {'Low Volatility': 0.05, 'Normal Volatility': 0.25,
                                'High Volatility': 0.65, 'Crisis Volatility': 0.95}
        economic_scenarios = {'Economic Growth': 0.15, 'Stable Economy': 0.0,
                              'Economic Decline': -0.15, 'Recession': -0.4}
    with col2:
        st.write('**Transaction Patterns**')
        frequency_multipliers = {
            'Normal Activity': 1.0, 'Increased Activity': 1.8, 'High Activity': 2.5, 'Reduced Activity': 0.3}
    if st.button('Run Scenario Analysis'):
        with st.spinner('Running scenario analysis...'):
            try:
                data = st.session_state.current_data.copy()
                predictor = st.session_state.predictor
                feature_engineer = st.session_state.feature_engineer
                scenario_results = {}
                for vol_name, vol_value in volatility_scenarios.items():
                    for econ_name, econ_value in economic_scenarios.items():
                        scenario_name = f'{vol_name} + {econ_name}'
                        scenario_data = data.copy()
                        scenario_data['market_volatility'] = vol_value
                        scenario_data['economic_indicator'] = econ_value
                        if 'Crisis' in vol_name or 'Recession' in econ_name:
                            scenario_data['transaction_frequency'] = scenario_data['transaction_frequency'] * 2.0
                            scenario_data['amount'] = scenario_data['amount'] * 1.3
                        elif 'Low' in vol_name and 'Growth' in econ_name:
                            scenario_data['transaction_frequency'] = scenario_data['transaction_frequency'] * 0.7
                            scenario_data['amount'] = scenario_data['amount'] * 0.9
                        processed_scenario = feature_engineer.transform(
                            scenario_data)
                        X_scenario = processed_scenario.drop(
                            columns=['risk_category', 'transaction_id'], errors='ignore')
                        scenario_scores = predictor.predict_risk_score(
                            X_scenario)
                        scenario_results[scenario_name] = scenario_scores
                st.subheader('Scenario Comparison')
                scenario_summary = {name: np.mean(
                    scores) for name, scores in scenario_results.items()}
                scenario_df = pd.DataFrame(list(scenario_summary.items()), columns=[
                                           'Scenario', 'Average Risk Score'])
                scenario_df = scenario_df.sort_values(
                    'Average Risk Score', ascending=False)
                viz = RiskVisualization()
                fig_scenario = px.bar(scenario_df, x='Scenario', y='Average Risk Score',
                                      title='Risk Scores Across Different Economic Scenarios', color='Average Risk Score', color_continuous_scale='RdYlBu_r')
                fig_scenario.update_layout(xaxis_tickangle=45)
                st.plotly_chart(fig_scenario, use_container_width=True)
                st.dataframe(scenario_df)
                st.subheader('Risk Distribution Changes')
                baseline_scores = scenario_results[list(
                    scenario_results.keys())[0]]
                worst_case_key = max(
                    scenario_summary, key=scenario_summary.get)
                worst_case_scores = scenario_results[worst_case_key]
                col1, col2 = st.columns(2)
                with col1:
                    st.metric('Baseline Avg Risk',
                              f'{np.mean(baseline_scores):.3f}')
                with col2:
                    st.metric('Worst Case Avg Risk',
                              f'{np.mean(worst_case_scores):.3f}')
            except Exception as e:
                st.error(f'Error running scenario analysis: {str(e)}')


if __name__ == '__main__':
    main()
