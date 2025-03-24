# Import our modules
from .data import CohesityDataProcessor
from .cost_optimizer import BackupCostOptimizer
from .predictive_models import BackupPredictionModels

import pandas as pd
import numpy as np
import json
from dash import no_update
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, dcc, html, dash_table
from dash.dependencies import Input, Output, State
from datetime import datetime
import base64
import io
import os



class BackupVisualizationDashboard:
    def __init__(self):
        """Initialize the visualization dashboard."""
        self.app = Dash(__name__, suppress_callback_exceptions=True)
        self.setup_layout()
        self.setup_callbacks()

    def setup_layout(self):
        """Set up the dashboard layout."""
        self.app.layout = html.Div([
            html.H1("Backup Cost Optimization Dashboard", style={"textAlign": "center", "marginTop": "20px"}),
            html.Div([
                html.Div([
                    html.H4("Data Upload"),
                    dcc.Upload(
                        id='upload-data',
                        children=html.Div(['Drag and Drop or ', html.A('Select a CSV File')]),
                        style={
                            'width': '100%', 'height': '60px', 'lineHeight': '60px',
                            'borderWidth': '1px', 'borderStyle': 'dashed',
                            'borderRadius': '5px', 'textAlign': 'center',
                            'margin': '10px'
                        },
                        multiple=False
                    ),
                    html.Div(id='upload-output')
                ], className="six columns"),
                html.Div([
                    html.H4("Uploaded Dataset Details"),
                    html.Div(id='dataset-info')
                ], className="six columns")
            ], className="row"),
            html.Hr(),
            html.Div([
                dcc.Graph(id='storage-growth-chart'),
                dcc.Graph(id='backup-job-growth-chart'),
                dcc.Graph(id='storage-forecast-chart')
            ]),
            html.Hr(),
            html.Div([
                dcc.Graph(id='cost-savings-chart'),
                dcc.Graph(id='tiering-recommendation-chart'),
                dcc.Graph(id='dedup-effectiveness-chart')
            ]),
            html.Hr(),
            dcc.Tabs([
                dcc.Tab(label='Retention Policies', children=[
                    dash_table.DataTable(id='retention-recommendations-table', page_size=10, export_format='csv')
                ]),
                dcc.Tab(label='Storage Tiering', children=[
                    dash_table.DataTable(id='tiering-recommendations-table', page_size=10, export_format='csv')
                ]),
                dcc.Tab(label='Deduplication Strategies', children=[
                    dash_table.DataTable(id='dedup-recommendations-table', page_size=10, export_format='csv')
                ]),
            ]),
            html.Hr(),
            html.Div([
                html.H2("Summary Report", style={"textAlign": "center"}),
                html.Div(id='summary-report'),
                html.Button("Download Full Report", id="download-report-button"),
                dcc.Download(id="download-report")
            ])
        ])

    def setup_callbacks(self):
        """Set up dashboard callbacks."""
        @self.app.callback(
            [Output('upload-output', 'children'),
             Output('dataset-info', 'children'),
             Output('storage-growth-chart', 'figure'),
             Output('backup-job-growth-chart', 'figure'),
             Output('storage-forecast-chart', 'figure'),
             Output('cost-savings-chart', 'figure'),
             Output('tiering-recommendation-chart', 'figure'),
             Output('dedup-effectiveness-chart', 'figure'),
             Output('retention-recommendations-table', 'data'),
             Output('retention-recommendations-table', 'columns'),
             Output('tiering-recommendations-table', 'data'),
             Output('tiering-recommendations-table', 'columns'),
             Output('dedup-recommendations-table', 'data'),
             Output('dedup-recommendations-table', 'columns'),
             Output('summary-report', 'children')],
            Input('upload-data', 'contents'),
            State('upload-data', 'filename')
        )
        def update_dashboard(contents, filename):
            if contents is None:
                empty_fig = go.Figure()
                empty_fig.update_layout(title="No Data")
                return ("No file uploaded yet.",
                        "No dataset available.",
                        empty_fig, empty_fig, empty_fig, empty_fig, empty_fig, empty_fig,
                        [], [{"name": "No Data", "id": "no_data"}],
                        [], [{"name": "No Data", "id": "no_data"}],
                        [], [{"name": "No Data", "id": "no_data"}],
                        html.Div("No data available."))
            # Decode uploaded CSV
            content_type, content_string = contents.split(',')
            decoded = base64.b64decode(content_string)
            df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
            # Process data using CohesityDataProcessor
            data_processor = CohesityDataProcessor(None)
            data_processor.df = df
            daily_summary = data_processor.aggregate_daily_backup_data()
            # Initialize cost optimizer and prediction models
            optimizer = BackupCostOptimizer()
            optimizer.df = df
            predictor = BackupPredictionModels()
            predictor.df = df
            # Prepare daily data for predictions; rename 'date' to 'backup_day'
            daily_data = daily_summary.copy()
            daily_data.rename(columns={'date': 'backup_day'}, inplace=True)
            # Run analyses
            backup_analysis = optimizer.analyze_backup_patterns()
            retention_recs = optimizer.recommend_retention_policies(backup_analysis)
            tiering_recs = optimizer.recommend_storage_tiering()
            dedup_recs = optimizer.recommend_deduplication_strategies(backup_analysis)
            predictions = predictor.predict_future_storage(daily_data, prediction_days=90, model_type='prophet')
            # Create plots
            storage_growth_fig = px.line(daily_summary, x='date', y='storage_used_GB', title='Storage Growth Over Time')
            backup_job_fig = px.line(daily_summary, x='date', y='backup_size_GB', title='Backup Size Over Time')
            forecast_fig = px.line(predictions, x='date', y='predicted_storage', title='Forecasted Storage Usage')
            cost_savings_fig = px.bar(daily_summary, x='date', y='cost_per_GB', title='Cost Per GB Over Time')
            tiering_fig = px.pie(tiering_recs['tier_recommendations'], names='suggested_tier', title='Storage Tiering Recommendations') if 'tier_recommendations' in tiering_recs else go.Figure()
            if dedup_recs.empty or 'job_name' not in dedup_recs.columns:
                dedup_fig = go.Figure()
                dedup_fig.update_layout(title="Deduplication Efficiency (No recommendations available)")
            else:
                dedup_fig = px.bar(dedup_recs, x='job_name', y='current_dedup_ratio', title='Deduplication Efficiency')            
            # Prepare tables
            retention_data = retention_recs.to_dict('records')
            retention_columns = [{"name": col, "id": col} for col in retention_recs.columns]
            if 'tier_recommendations' in tiering_recs:
                tiering_df = tiering_recs['tier_recommendations']
                if not tiering_df.empty:
                    tiering_data = tiering_df.to_dict('records')
                    tiering_columns = [{"name": col, "id": col} for col in tiering_df.columns]
                else:
                    tiering_data = []
                    tiering_columns = []
            else:
                tiering_data = []
                tiering_columns = []
            if not dedup_recs.empty and 'job_name' in dedup_recs.columns:
                dedup_data = dedup_recs.to_dict('records')
                dedup_columns = [{"name": col, "id": col} for col in dedup_recs.columns]
            else:
                dedup_data = []
                dedup_columns = []
            summary_report = html.Div([
                html.P(f"Total Backup Jobs: {df.shape[0]}"),
                html.P(f"Total Storage Used: {df['storage_used_GB'].sum()} GB"),
                html.P(f"Average Deduplication Ratio: {df['dedup_ratio'].mean():.2f}")
            ])
            return (html.Div(f"File '{filename}' processed successfully."),
                    html.Div(f"Dataset contains {df.shape[0]} rows and {df.shape[1]} columns."),
                    storage_growth_fig,
                    backup_job_fig,
                    forecast_fig,
                    cost_savings_fig,
                    tiering_fig,
                    dedup_fig,
                    retention_data,
                    retention_columns,
                    tiering_data,
                    tiering_columns,
                    dedup_data,
                    dedup_columns,
                    summary_report)

        # Callback for downloading the full report
        @self.app.callback(
            Output("download-report", "data"),
            Input("download-report-button", "n_clicks"),
            State('upload-data', 'contents'),
            State('upload-data', 'filename'),
            prevent_initial_call=True
        )
        def generate_full_report(n_clicks, contents, filename):
            if contents is None:
                return no_update
            # Decode uploaded CSV
            content_type, content_string = contents.split(',')
            decoded = base64.b64decode(content_string)
            df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
            
            # Initialize the cost optimizer to generate the comprehensive report
            optimizer = BackupCostOptimizer()
            optimizer.df = df
            report = optimizer.generate_comprehensive_report()
            
            # Create a CSV file with multiple sections
            csv_buffer = io.StringIO()
            
            # Section 1: Summary
            csv_buffer.write("SUMMARY\n")
            summary_df = pd.DataFrame([report['summary']])
            csv_buffer.write(summary_df.to_csv(index=False))
            csv_buffer.write("\n")
            
            # Section 2: Retention Recommendations
            csv_buffer.write("RETENTION RECOMMENDATIONS\n")
            if not report['retention_recommendations'].empty:
                csv_buffer.write(report['retention_recommendations'].to_csv(index=False))
            else:
                csv_buffer.write("No retention recommendations available.\n")
            csv_buffer.write("\n")
            
            # Section 3: Tiering Recommendations
            csv_buffer.write("STORAGE TIERING RECOMMENDATIONS\n")
            # tiering_recommendations is a dict with key 'tier_recommendations'
            if 'tier_recommendations' in report['tiering_recommendations']:
                tiering_df = report['tiering_recommendations']['tier_recommendations']
                if not tiering_df.empty:
                    csv_buffer.write(tiering_df.to_csv(index=False))
                else:
                    csv_buffer.write("No tiering recommendations available.\n")
            else:
                csv_buffer.write("No tiering recommendations available.\n")
            csv_buffer.write("\n")
            
            # Section 4: Deduplication Recommendations
            csv_buffer.write("DEDUPLICATION RECOMMENDATIONS\n")
            if not report['dedup_recommendations'].empty:
                csv_buffer.write(report['dedup_recommendations'].to_csv(index=False))
            else:
                csv_buffer.write("No deduplication recommendations available.\n")
            
            return dict(content=csv_buffer.getvalue(), filename="full_report.csv", type="text/csv")        

    def run(self):
        """Run the dashboard."""
        self.app.run(debug=True)

if __name__ == "__main__":
    dashboard = BackupVisualizationDashboard()
    dashboard.run()
