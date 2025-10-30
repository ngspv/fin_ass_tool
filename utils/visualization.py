import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


class RiskVisualization:

    def __init__(self):
        self.color_scheme = {'Low': '#2E8B57',
                             'Medium': '#FF8C00', 'High': '#DC143C'}

    def plot_risk_distribution(self, df, title='Risk Category Distribution'):
        risk_counts = df['risk_category'].value_counts()
        fig = px.pie(values=risk_counts.values, names=risk_counts.index, title=title,
                     color=risk_counts.index, color_discrete_map=self.color_scheme)
        fig.update_traces(textposition='inside', textinfo='percent+label')
        fig.update_layout(font=dict(size=14),
                          title_font_size=16, showlegend=True)
        return fig

    def plot_amount_vs_risk(self, df, title='Transaction Amount vs Risk Score'):
        fig = px.scatter(df, x='amount', y='risk_score', color='risk_category', size='transaction_frequency', hover_data=[
                         'asset_type', 'user_segment'], title=title, color_discrete_map=self.color_scheme, log_x=True)
        fig.update_layout(xaxis_title='Transaction Amount (Log Scale)',
                          yaxis_title='Risk Score', font=dict(size=12), title_font_size=16)
        return fig

    def plot_risk_by_asset_type(self, df, title='Risk Distribution by Asset Type'):
        risk_by_asset = pd.crosstab(
            df['asset_type'], df['risk_category'], normalize='index') * 100
        fig = go.Figure()
        for risk_level in ['Low', 'Medium', 'High']:
            if risk_level in risk_by_asset.columns:
                fig.add_trace(go.Bar(name=risk_level, x=risk_by_asset.index,
                              y=risk_by_asset[risk_level], marker_color=self.color_scheme[risk_level]))
        fig.update_layout(barmode='stack', title=title, xaxis_title='Asset Type', yaxis_title='Percentage', font=dict(
            size=12), title_font_size=16, legend=dict(title='Risk Category'))
        return fig

    def plot_risk_timeline(self, df, title='Risk Trends Over Time'):
        df_time = df.copy()
        df_time['transaction_date'] = pd.to_datetime(
            df_time['transaction_date'])
        df_time['year_month'] = df_time['transaction_date'].dt.to_period('M')
        monthly_risk = df_time.groupby(
            ['year_month', 'risk_category']).size().unstack(fill_value=0)
        monthly_risk.index = monthly_risk.index.to_timestamp()
        fig = go.Figure()
        for risk_level in ['Low', 'Medium', 'High']:
            if risk_level in monthly_risk.columns:
                fig.add_trace(go.Scatter(x=monthly_risk.index, y=monthly_risk[risk_level], mode='lines+markers', name=f'{risk_level} Risk', line=dict(
                    color=self.color_scheme[risk_level], width=3), marker=dict(size=6)))
        fig.update_layout(title=title, xaxis_title='Date', yaxis_title='Number of Transactions', font=dict(
            size=12), title_font_size=16, hovermode='x unified')
        return fig

    def plot_feature_importance(self, feature_importance_df, top_n=15, title='Top Feature Importance'):
        top_features = feature_importance_df.head(top_n)
        fig = px.bar(top_features, x='importance', y='feature', orientation='h',
                     title=title, color='importance', color_continuous_scale='viridis')
        fig.update_layout(yaxis={'categoryorder': 'total ascending'}, xaxis_title='Importance Score',
                          yaxis_title='Features', font=dict(size=12), title_font_size=16, height=600)
        return fig

    def plot_correlation_heatmap(self, df, title='Feature Correlation Heatmap'):
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        corr_matrix = df[numerical_cols].corr()
        fig = px.imshow(corr_matrix, text_auto=True, aspect='auto',
                        color_continuous_scale='RdBu', title=title)
        fig.update_layout(font=dict(size=10), title_font_size=16, height=600)
        return fig

    def plot_risk_by_user_segment(self, df, title='Risk Distribution by User Segment'):
        segment_risk = pd.crosstab(df['user_segment'], df['risk_category'])
        fig = go.Figure()
        for risk_level in ['Low', 'Medium', 'High']:
            if risk_level in segment_risk.columns:
                fig.add_trace(go.Bar(name=risk_level, x=segment_risk.index,
                              y=segment_risk[risk_level], marker_color=self.color_scheme[risk_level]))
        fig.update_layout(barmode='group', title=title, xaxis_title='User Segment', yaxis_title='Number of Transactions', font=dict(
            size=12), title_font_size=16, legend=dict(title='Risk Category'))
        return fig

    def plot_volatility_impact(self, df, title='Market Volatility vs Risk Score'):
        df_vol = df.copy()
        df_vol['volatility_category'] = pd.cut(df_vol['market_volatility'], bins=[
                                               0, 0.2, 0.4, 0.6, 1.0], labels=['Low', 'Medium', 'High', 'Very High'])
        fig = px.box(df_vol, x='volatility_category', y='risk_score', color='volatility_category',
                     title=title, color_discrete_sequence=['#2E8B57', '#FF8C00', '#DC143C', '#8B0000'])
        fig.update_layout(xaxis_title='Market Volatility Category', yaxis_title='Risk Score', font=dict(
            size=12), title_font_size=16, showlegend=False)
        return fig

    def create_risk_dashboard(self, df, feature_importance_df=None):
        dashboard = {}
        dashboard['risk_distribution'] = self.plot_risk_distribution(df)
        dashboard['amount_vs_risk'] = self.plot_amount_vs_risk(df)
        dashboard['risk_by_asset'] = self.plot_risk_by_asset_type(df)
        dashboard['risk_by_segment'] = self.plot_risk_by_user_segment(df)
        dashboard['risk_timeline'] = self.plot_risk_timeline(df)
        dashboard['volatility_impact'] = self.plot_volatility_impact(df)
        if feature_importance_df is not None:
            dashboard['feature_importance'] = self.plot_feature_importance(
                feature_importance_df)
        return dashboard

    def plot_scenario_analysis(self, scenario_results, title='Scenario Analysis Results'):
        scenarios = list(scenario_results.keys())
        risk_scores = [np.mean(scores) for scores in scenario_results.values()]
        fig = px.bar(x=scenarios, y=risk_scores, title=title,
                     color=risk_scores, color_continuous_scale='RdYlBu_r')
        fig.update_layout(xaxis_title='Scenarios', yaxis_title='Average Risk Score', font=dict(
            size=12), title_font_size=16, showlegend=False)
        if 'baseline' in scenarios:
            baseline_score = risk_scores[scenarios.index('baseline')]
            fig.add_hline(y=baseline_score, line_dash='dash',
                          line_color='red', annotation_text='Baseline')
        return fig


def create_risk_summary_stats(df):
    stats = {'total_transactions': len(df), 'high_risk_count': len(df[df['risk_category'] == 'High']), 'high_risk_percentage': len(df[df['risk_category'] == 'High']) / len(df) * 100, 'average_risk_score': df['risk_score'].mean(
    ), 'max_risk_score': df['risk_score'].max(), 'high_value_transactions': len(df[df['amount'] > df['amount'].quantile(0.9)]), 'weekend_transactions': len(df[df['is_weekend'] == True]), 'after_hours_transactions': len(df[df['is_after_hours'] == True])}
    return stats


if __name__ == '__main__':
    from utils.data_generator import generate_transaction_data
    df = generate_transaction_data(1000)
    viz = RiskVisualization()
    risk_dist_fig = viz.plot_risk_distribution(df)
    amount_risk_fig = viz.plot_amount_vs_risk(df)
    dashboard = viz.create_risk_dashboard(df)
    print('Visualizations created successfully!')
    print(f'Dashboard contains {len(dashboard)} charts')
