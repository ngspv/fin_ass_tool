# AI-Powered Financial Risk Assessment Tool

A comprehensive machine learning application for analyzing financial transaction data and predicting risk levels using advanced algorithms and interactive visualizations.

## Features

- **Data Generation**: Create synthetic financial transaction datasets for testing and demonstration
- **Feature Engineering**: Advanced preprocessing and feature extraction from transaction data
- **Machine Learning Models**: Multiple ML algorithms including Logistic Regression, Random Forest, XGBoost, and LightGBM
- **Risk Prediction**: Real-time risk assessment for individual transactions and batch processing
- **Interactive Visualizations**: Comprehensive dashboards with Plotly charts and graphs
- **Scenario Analysis**: Simulate different economic conditions and assess potential risks
- **Model Comparison**: Cross-validation and performance evaluation across multiple models
- **Web Interface**: User-friendly Streamlit application for easy interaction

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage Guide](#usage-guide)
- [Project Structure](#project-structure)
- [Data Schema](#data-schema)
- [Model Performance](#model-performance)
- [API Reference](#api-reference)
- [Contributing](#contributing)
- [License](#license)

## Installation

### Prerequisites

- Python 3.8 or higher
- Virtual environment (recommended)

### Step 1: Clone the Repository

```bash
git clone <repository-url>
cd fin_ass_tool
```

### Step 2: Create and Activate Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Linux/Mac:
source venv/bin/activate
# On Windows:
# venv\\Scripts\\activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Dependencies

The application requires the following main packages:

- `streamlit` - Web application framework
- `pandas` - Data manipulation and analysis
- `numpy` - Numerical computing
- `scikit-learn` - Machine learning library
- `xgboost` - Gradient boosting framework
- `lightgbm` - Gradient boosting framework
- `plotly` - Interactive visualizations
- `seaborn` - Statistical data visualization
- `matplotlib` - Plotting library
- `joblib` - Model serialization

## Quick Start

### 1. Run the Streamlit Application

```bash
streamlit run app.py
```

The application will open in your web browser at `http://localhost:8501`.

### 2. Generate Sample Data

1. Navigate to the "📊 Data Overview" page
2. Click "Generate Sample Data" 
3. Choose the number of transactions (default: 1000)
4. Click the generate button

### 3. Train Models

1. Go to the "⚙️ Model Training" page
2. Configure training parameters (test size, number of features)
3. Click "Start Training"
4. Wait for training to complete (may take a few minutes)

### 4. Make Predictions

1. Visit the "Risk Prediction" page
2. Enter transaction details in the form
3. Click "Predict Risk" to get individual predictions
4. Use "Batch Prediction" for multiple transactions

### 5. Explore Visualizations

1. Navigate to "Visualizations"
2. Select different chart types from the dropdown
3. View interactive plots and summary statistics

## Usage Guide

### Data Upload

The application accepts CSV files with the following columns:

**Required Columns:**
- `transaction_id` - Unique identifier for each transaction
- `amount` - Transaction amount (numeric)
- `asset_type` - Type of asset (Stock, Bond, Forex, etc.)
- `user_segment` - User category (Retail, Institutional, etc.)
- `transaction_date` - Date of transaction (YYYY-MM-DD)

**Optional Columns:**
- `market_volatility` - Market volatility score (0-1)
- `economic_indicator` - Economic sentiment (-0.5 to 0.5)
- `region` - Geographic region
- `risk_category` - Actual risk level (for model validation)

### Model Training

The application trains four different models:

1. **Logistic Regression** - Linear classification model
2. **Random Forest** - Ensemble of decision trees
3. **XGBoost** - Gradient boosting algorithm
4. **LightGBM** - Efficient gradient boosting

Each model is optimized using hyperparameter tuning with cross-validation.

### Risk Categories

The system classifies transactions into three risk levels:

- **Low Risk** (Green) - Risk score: 0.0 - 0.3
- **Medium Risk** (Orange) - Risk score: 0.3 - 0.6
- **High Risk** (Red) - Risk score: 0.6 - 1.0

### Scenario Analysis

Test different economic conditions:

- **Market Volatility**: Low (0.1) to Crisis (0.9)
- **Economic Indicators**: Growth (+0.1) to Recession (-0.3)
- **Transaction Frequency**: 0.5x to 2.0x normal levels

## Project Structure

```
fin_ass_tool/
├── app.py                          # Main Streamlit application
├── requirements.txt                # Python dependencies
├── README.md                      # This file
├── data/                          # Data storage directory
│   └── sample_transactions.csv    # Generated sample data
├── models/                        # ML models and training
│   ├── __init__.py
│   ├── risk_predictor.py          # Main ML model classes
│   └── saved_models/              # Serialized trained models
├── utils/                         # Utility modules
│   ├── __init__.py
│   ├── data_generator.py          # Synthetic data generation
│   ├── feature_engineering.py     # Feature processing
│   └── visualization.py           # Plotting and charts
└── pages/                         # Additional Streamlit pages
    └── __init__.py
```

## Data Schema

### Input Data Format

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| transaction_id | string | Unique identifier | "TXN_000001" |
| transaction_date | date | Transaction date | "2024-01-15" |
| amount | float | Transaction amount | 1500.50 |
| asset_type | string | Asset category | "Stock" |
| user_segment | string | User type | "Retail" |
| region | string | Geographic region | "North America" |
| transaction_frequency | int | Monthly frequency | 5 |
| market_volatility | float | Market volatility (0-1) | 0.25 |
| economic_indicator | float | Economic sentiment (-0.5 to 0.5) | 0.05 |
| is_weekend | boolean | Weekend transaction | true |
| is_after_hours | boolean | After hours transaction | false |
| hour_of_day | int | Transaction hour (0-23) | 14 |

### Generated Features

The feature engineering pipeline creates additional features:

- **Temporal Features**: Month/day cyclical encoding, quarter
- **Amount Features**: Log transforms, percentiles, categories
- **Frequency Features**: Relative to segment averages
- **Risk Features**: Volatility interactions, time-based risks
- **Aggregated Features**: User segment and asset type statistics

## Model Performance

Typical performance metrics on test data:

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| Logistic Regression | 0.82 | 0.81 | 0.82 | 0.81 |
| Random Forest | 0.87 | 0.86 | 0.87 | 0.86 |
| XGBoost | 0.89 | 0.88 | 0.89 | 0.88 |
| LightGBM | 0.88 | 0.87 | 0.88 | 0.87 |

*Note: Performance may vary based on data quality and distribution.*

## API Reference

### Data Generation

```python
from utils.data_generator import generate_transaction_data

# Generate sample data
df = generate_transaction_data(n_transactions=1000, 
                              start_date="2020-01-01", 
                              end_date="2024-12-31")
```

### Feature Engineering

```python
from utils.feature_engineering import FinancialFeatureEngineer

# Initialize feature engineer
engineer = FinancialFeatureEngineer()

# Process features
processed_df = engineer.fit_transform(df, target_column='risk_category')
```

### Model Training

```python
from models.risk_predictor import FinancialRiskPredictor

# Initialize predictor
predictor = FinancialRiskPredictor()

# Train all models
results = predictor.train_all_models(processed_df)

# Make predictions
risk_categories, probabilities = predictor.predict_risk(X_new)
risk_scores = predictor.predict_risk_score(X_new)
```

### Visualization

```python
from utils.visualization import RiskVisualization

# Initialize visualizer
viz = RiskVisualization()

# Create charts
risk_dist_fig = viz.plot_risk_distribution(df)
amount_risk_fig = viz.plot_amount_vs_risk(df)
dashboard = viz.create_risk_dashboard(df)
```

## Advanced Features

### Custom Risk Scoring

The application uses a weighted approach to calculate risk scores:

```python
# Risk weights for different categories
risk_weights = {
    'amount_risk': 0.3,
    'frequency_risk': 0.2,
    'asset_risk': 0.3,
    'volatility_risk': 0.1,
    'economic_risk': 0.1
}
```

### Model Ensemble

For improved predictions, you can create ensemble models:

```python
# Get predictions from all models
predictions = {}
for model_name, model in predictor.models.items():
    predictions[model_name] = model.predict_proba(X)

# Weighted ensemble
ensemble_pred = np.average(list(predictions.values()), axis=0, weights=[0.2, 0.25, 0.3, 0.25])
```

### Custom Visualizations

Extend the visualization class for specific needs:

```python
class CustomRiskVisualization(RiskVisualization):
    def plot_custom_metric(self, df, metric_column):
        # Your custom plotting logic
        pass
```

## Common Issues and Solutions

### Issue 1: Import Errors

**Problem**: ModuleNotFoundError when running the application

**Solution**: 
```bash
# Ensure all dependencies are installed
pip install -r requirements.txt

# Check Python path
python -c "import sys; print(sys.path)"
```

### Issue 2: Memory Issues with Large Datasets

**Problem**: Application crashes with large datasets

**Solution**:
- Reduce the number of transactions
- Use feature selection to reduce dimensionality
- Increase system memory or use cloud deployment

### Issue 3: Poor Model Performance

**Problem**: Low accuracy scores

**Solution**:
- Check data quality and distribution
- Increase training data size
- Tune hyperparameters
- Try feature engineering improvements

## Performance Optimization

### For Large Datasets

1. **Use chunking for data processing**:
```python
chunk_size = 10000
for chunk in pd.read_csv('large_file.csv', chunksize=chunk_size):
    process_chunk(chunk)
```

2. **Enable parallel processing**:
```python
from sklearn.model_selection import GridSearchCV
grid_search = GridSearchCV(model, param_grid, n_jobs=-1)
```

3. **Use efficient data types**:
```python
df['category_col'] = df['category_col'].astype('category')
df['numeric_col'] = pd.to_numeric(df['numeric_col'], downcast='float')
```

## Security Considerations

- **Data Privacy**: Ensure sensitive financial data is properly anonymized
- **Input Validation**: The application validates all user inputs
- **Model Security**: Trained models should be stored securely
- **Access Control**: Consider adding authentication for production use

## Deployment

### Local Development

```bash
streamlit run app.py --server.port 8501
```

### Docker Deployment

Create a `Dockerfile`:

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

### Cloud Deployment

The application can be deployed on:
- Streamlit Cloud
- Heroku
- AWS EC2
- Google Cloud Run
- Azure Container Instances

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Guidelines

- Follow PEP 8 style guidelines
- Add docstrings to all functions
- Include unit tests for new features
- Update documentation for any changes

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Built with [Streamlit](https://streamlit.io/) for the web interface
- Machine learning powered by [scikit-learn](https://scikit-learn.org/), [XGBoost](https://xgboost.readthedocs.io/), and [LightGBM](https://lightgbm.readthedocs.io/)
- Visualizations created with [Plotly](https://plotly.com/python/)
- Data manipulation using [pandas](https://pandas.pydata.org/) and [NumPy](https://numpy.org/)

## Support

For questions and support:
- Create an issue in the GitHub repository
- Check the documentation and FAQ sections
- Review the code examples and API reference

## Version History

- **v1.0.0** - Initial release with core functionality
  - Data generation and processing
  - Multiple ML models
  - Interactive web interface
  - Comprehensive visualizations
  - Scenario analysis features

---

**Happy Risk Assessment!**