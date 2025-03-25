import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import List, Dict, Any, Tuple, Optional, Union
import logging
from utils import detect_column_types

logger = logging.getLogger("csv_cleaner")

class DataVisualizer:
    """
    A class for visualizing data quality and distributions in a pandas DataFrame.
    Provides interactive visualizations for missing values, outliers, and distributions.
    """
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialize the DataVisualizer with a pandas DataFrame.
        
        Args:
            df: The pandas DataFrame to visualize
        """
        self.df = df.copy()
        self.column_types = detect_column_types(self.df)
        self.numeric_cols = [col for col, type_ in self.column_types.items() 
                           if type_ in ['numeric', 'integer']]
        self.categorical_cols = [col for col, type_ in self.column_types.items() 
                               if type_ in ['categorical', 'boolean']]
        self.datetime_cols = [col for col, type_ in self.column_types.items() 
                            if type_ == 'datetime']
        
    def _is_datetime(self, series: pd.Series) -> bool:
        """Check if a series can be converted to datetime."""
        if series.dtype == 'datetime64[ns]':
            return True
            
        if series.dtype == 'object':
            try:
                pd.to_datetime(series.dropna().iloc[0:5])
                return True
            except:
                return False
                
        return False
        
    def plot_missing_values(self) -> Dict[str, Any]:
        """
        Create visualizations for missing values in the dataset.
            
        Returns:
            Dictionary with missing values visualizations
        """
        # Get missing value counts by column
        missing_values = self.df.isna().sum()
        missing_percent = missing_values / len(self.df) * 100
        
        # Create a DataFrame for visualization
        missing_df = pd.DataFrame({
            'Column': missing_values.index,
            'Missing Count': missing_values.values,
            'Missing Percentage': missing_percent.values
        })
        missing_df = missing_df.sort_values('Missing Percentage', ascending=False)
        
        # Only keep columns with missing values
        missing_df = missing_df[missing_df['Missing Count'] > 0]
        
        if missing_df.empty:
            return {
                'has_missing': False,
                'message': "No missing values found in the dataset.",
                'plots': {}
            }
        
        # Create bar chart for missing values
        fig_bar = px.bar(
            missing_df, 
            x='Column', 
            y='Missing Percentage',
            title='Percentage of Missing Values by Column',
            color='Missing Percentage',
            color_continuous_scale=['#2ecc71', '#f39c12', '#e74c3c'],
            text='Missing Count',
            height=400
        )
        fig_bar.update_traces(
            texttemplate='%{text}',
            textposition='outside',
            hovertemplate='<b>%{x}</b><br>Missing: %{text}<br>Percentage: %{y:.2f}%'
        )
        fig_bar.update_yaxes(title='Missing Percentage (%)', range=[0, 100])
        fig_bar.update_layout(xaxis_tickangle=-45)
        
        # Create heatmap for missing value patterns
        cols_with_missing = missing_df['Column'].tolist()
        
        if len(cols_with_missing) > 0:
            # Sample rows for the heatmap (to avoid large visualizations)
            sample_size = min(1000, len(self.df))
            df_sample = self.df.sample(sample_size) if len(self.df) > sample_size else self.df
            
            # Create missing value mask
            missing_mask = df_sample[cols_with_missing].isna()
            
            # Create heatmap
            fig_heatmap = px.imshow(
                missing_mask.T,
                labels=dict(x="Row Index", y="Column", color="Is Missing"),
                title='Missing Value Patterns (Sample of Data)',
                color_continuous_scale=['#ffffff', '#e74c3c'],
                aspect="auto",
                height=400
            )
            fig_heatmap.update_xaxes(showticklabels=False)
            
            # Calculate correlation of missingness
            if len(cols_with_missing) > 1:
                corr_matrix = missing_mask.corr()
                
                fig_corr = px.imshow(
                    corr_matrix,
                    labels=dict(x="Column", y="Column", color="Correlation"),
                    title='Correlation of Missing Values Between Columns',
                    color_continuous_scale=['#3498db', '#ffffff', '#e74c3c'],
                    zmin=-1, zmax=1,
                    height=500
                )
                
                return {
                    'has_missing': True,
                    'message': f"Found {len(cols_with_missing)} columns with missing values.",
                    'missing_data': missing_df.to_dict(orient='records'),
                    'plots': {
                        'bar_chart': fig_bar,
                        'heatmap': fig_heatmap,
                        'correlation': fig_corr
                    }
                }
            
        return {
            'has_missing': True,
            'message': f"Found {len(cols_with_missing)} columns with missing values.",
            'missing_data': missing_df.to_dict(orient='records'),
            'plots': {
                'bar_chart': fig_bar,
                'heatmap': fig_heatmap
            }
        }
        
    def plot_outliers(self, column: str, method: str = 'iqr', threshold: float = 1.5) -> Dict[str, Any]:
        """
        Create visualizations for outliers in a specified column.
        
        Args:
            column: Column to analyze for outliers
            method: Method to use for outlier detection ('iqr' or 'zscore')
            threshold: Threshold for outlier detection
            
        Returns:
            Dictionary with outlier visualizations
        """
        # Check if column exists and is numeric
        if column not in self.df.columns:
            return {'error': f"Column '{column}' not found in the dataset."}
            
        if column not in self.numeric_cols:
            return {'error': f"Column '{column}' is not numeric. Cannot detect outliers."}
            
        # Get column data, excluding missing values
        col_data = self.df[column].dropna()
        
        if len(col_data) == 0:
            return {'error': f"Column '{column}' has no non-null values."}
            
        # Calculate bounds based on the chosen method
        if method == 'iqr':
            Q1 = col_data.quantile(0.25)
            Q3 = col_data.quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            
            # Identify outliers
            outliers_mask = (col_data < lower_bound) | (col_data > upper_bound)
            
        elif method == 'zscore':
            # Apply Z-score method
            from scipy import stats
            z_scores = stats.zscore(col_data, nan_policy='omit')
            outliers_mask = abs(z_scores) > threshold
            
            # Calculate approximate bounds for visualization
            mean = col_data.mean()
            std = col_data.std()
            lower_bound = mean - threshold * std
            upper_bound = mean + threshold * std
        else:
            return {'error': "Method must be either 'iqr' or 'zscore'"}
        
        # Count outliers
        outlier_count = outliers_mask.sum()
        outlier_percent = (outlier_count / len(col_data)) * 100
        
        # Create a DataFrame for the box plot
        box_data = pd.DataFrame({column: col_data})
        box_data['is_outlier'] = outliers_mask
        
        # Create box plot
        fig_box = px.box(
            box_data, 
            y=column,
            title=f'Box Plot of {column} with Outliers',
            points='all',  # show all points
            color='is_outlier',
            color_discrete_map={True: '#e74c3c', False: '#3498db'},
            labels={'is_outlier': 'Is Outlier'},
            height=500
        )
        
        # Create histogram with outlier bounds
        fig_hist = px.histogram(
            box_data,
            x=column, 
            color='is_outlier',
            color_discrete_map={True: '#e74c3c', False: '#3498db'},
            marginal='rug',
            title=f'Distribution of {column} with Outlier Thresholds',
            labels={'is_outlier': 'Is Outlier'},
            height=500,
            opacity=0.7
        )
        
        # Add vertical lines for bounds
        fig_hist.add_vline(x=lower_bound, line_dash="dash", line_color="red", annotation_text="Lower Bound")
        fig_hist.add_vline(x=upper_bound, line_dash="dash", line_color="red", annotation_text="Upper Bound")
        
        # Create QQ plot for normality check
        from scipy import stats
        qq_x = np.arange(0.01, 0.99, 0.01)
        qq_y = np.quantile(col_data, qq_x)
        
        theoretical_quantiles = stats.norm.ppf(qq_x, loc=col_data.mean(), scale=col_data.std())
        
        fig_qq = go.Figure()
        fig_qq.add_trace(go.Scatter(
            x=theoretical_quantiles,
            y=qq_y,
            mode='markers',
            marker=dict(color='#3498db'),
            name='QQ Plot'
        ))
        
        # Add the 45-degree reference line
        min_val = min(min(theoretical_quantiles), min(qq_y))
        max_val = max(max(theoretical_quantiles), max(qq_y))
        fig_qq.add_trace(go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode='lines',
            line=dict(color='red', dash='dash'),
            name='Reference Line'
        ))
        
        fig_qq.update_layout(
            title=f'QQ Plot for {column}',
            xaxis_title='Theoretical Quantiles',
            yaxis_title='Sample Quantiles',
            height=500
        )
        
        return {
            'has_outliers': outlier_count > 0,
            'outlier_count': int(outlier_count),
            'outlier_percent': round(outlier_percent, 2),
            'stats': {
                'mean': float(col_data.mean()),
                'median': float(col_data.median()),
                'std': float(col_data.std()),
                'min': float(col_data.min()),
                'max': float(col_data.max()),
                'lower_bound': float(lower_bound),
                'upper_bound': float(upper_bound)
            },
            'plots': {
                'box_plot': fig_box,
                'histogram': fig_hist,
                'qq_plot': fig_qq
            }
        }
        
    def plot_distribution(self, column: str) -> Dict[str, Any]:
        """
        Create visualizations for the distribution of a specified column.
        
        Args:
            column: Column to visualize
            
        Returns:
            Dictionary with distribution visualizations
        """
        # Check if column exists
        if column not in self.df.columns:
            return {'error': f"Column '{column}' not found in the dataset."}
            
        # Get column type
        col_type = self.column_types.get(column, 'unknown')
        
        # Create different visualizations based on column type
        if col_type in ['numeric', 'integer']:
            # Create histogram for numeric data
            fig_hist = px.histogram(
                self.df,
                x=column,
                title=f'Distribution of {column}',
                marginal='box',
                height=400,
                color_discrete_sequence=['#3498db']
            )
            
            # Add mean and median lines
            mean_val = self.df[column].mean()
            median_val = self.df[column].median()
            
            fig_hist.add_vline(x=mean_val, line_dash="solid", line_color="#e74c3c", 
                            annotation_text=f"Mean: {mean_val:.2f}")
            fig_hist.add_vline(x=median_val, line_dash="dash", line_color="#2ecc71", 
                            annotation_text=f"Median: {median_val:.2f}")
            
            # Create violin plot
            fig_violin = px.violin(
                self.df,
            y=column,
                title=f'Violin Plot of {column}',
                box=True,
                points='all',
                height=400,
                color_discrete_sequence=['#9b59b6']
            )
            
            return {
                'column_type': col_type,
                'plots': {
                    'histogram': fig_hist,
                    'violin': fig_violin
                },
                'stats': {
                    'mean': float(mean_val),
                    'median': float(median_val),
                    'std': float(self.df[column].std()),
                    'min': float(self.df[column].min()),
                    'max': float(self.df[column].max()),
                    'missing': int(self.df[column].isna().sum())
                }
            }
            
        elif col_type in ['categorical', 'boolean']:
            # Create bar chart for categorical data
            value_counts = self.df[column].value_counts().reset_index()
            value_counts.columns = [column, 'Count']
            
            # Limit to top 20 categories if there are many
            if len(value_counts) > 20:
                other_count = value_counts.iloc[20:]['Count'].sum()
                value_counts = value_counts.iloc[:20]
                
                # Add an "Other" category for the rest
                other_row = pd.DataFrame({column: ['Other'], 'Count': [other_count]})
                value_counts = pd.concat([value_counts, other_row], ignore_index=True)
            
            fig_bar = px.bar(
                value_counts,
                x=column,
                y='Count',
                title=f'Distribution of {column}',
                color_discrete_sequence=['#3498db'],
                text='Count',
                height=400
            )
            fig_bar.update_xaxes(tickangle=-45)
            
            # Create pie chart
            fig_pie = px.pie(
                value_counts,
                names=column,
                values='Count',
                title=f'Distribution of {column}',
                height=400
            )
            
            return {
                'column_type': col_type,
                'plots': {
                    'bar_chart': fig_bar,
                    'pie_chart': fig_pie
                },
                'stats': {
                    'unique_values': int(self.df[column].nunique()),
                    'top_value': str(self.df[column].mode()[0]) if not self.df[column].mode().empty else None,
                    'missing': int(self.df[column].isna().sum())
                }
            }
            
        elif col_type == 'datetime':
            # Convert to datetime if needed
            date_series = pd.to_datetime(self.df[column])
            
            # Create time series histogram
            fig_time = px.histogram(
                self.df,
                x=date_series,
                title=f'Distribution of {column} Over Time',
                color_discrete_sequence=['#3498db'],
                height=400
            )
            
            # Create box plot by month
            df_time = pd.DataFrame({
                'date': date_series,
                'month': date_series.dt.month,
                'year': date_series.dt.year,
                'day': date_series.dt.day,
                'weekday': date_series.dt.weekday
            })
            
            fig_month = px.box(
                df_time,
                x='month',
                y='day',
                title=f'Distribution of {column} by Month',
                color_discrete_sequence=['#9b59b6'],
                height=400
            )
            
            return {
                'column_type': 'datetime',
                'plots': {
                    'time_histogram': fig_time,
                    'month_distribution': fig_month
                },
                'stats': {
                    'min_date': str(date_series.min()),
                    'max_date': str(date_series.max()),
                    'missing': int(self.df[column].isna().sum())
                }
            }
            
        else:  # text or other types
            # For text, show length distribution
            text_lengths = self.df[column].astype(str).apply(len)
            
            fig_hist = px.histogram(
                text_lengths,
                title=f'Distribution of Text Length in {column}',
                labels={'value': 'Text Length'},
                color_discrete_sequence=['#3498db'],
                height=400
            )
            
            # Create a sample of values
            sample_values = self.df[column].dropna().sample(min(5, self.df[column].dropna().shape[0])).tolist()
            
            return {
                'column_type': 'text',
                'plots': {
                    'length_histogram': fig_hist
                },
                'stats': {
                    'avg_length': float(text_lengths.mean()),
                    'max_length': float(text_lengths.max()),
                    'missing': int(self.df[column].isna().sum()),
                    'sample_values': sample_values
                }
            }
    
    def compare_before_after(self, column: str, original_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Compare a column's distribution before and after cleaning.
        
        Args:
            column: Column to compare
            original_df: Original DataFrame before cleaning
            
        Returns:
            Dictionary with comparison visualizations
        """
        # Check if column exists in both DataFrames
        if column not in self.df.columns or column not in original_df.columns:
            return {'error': f"Column '{column}' not found in both datasets."}
            
        # Get column type
        col_type = self.column_types.get(column, 'unknown')
        
        # Create comparison visualizations based on column type
        if col_type in ['numeric', 'integer']:
            # Create histogram overlay
            fig = go.Figure()
            
            fig.add_trace(go.Histogram(
                x=original_df[column],
                name='Before',
                opacity=0.5,
                marker_color='#3498db'
            ))
            
            fig.add_trace(go.Histogram(
                x=self.df[column],
                name='After',
                opacity=0.5,
                marker_color='#e74c3c'
            ))
            
            fig.update_layout(
                title_text=f'Distribution of {column} Before and After Cleaning',
                xaxis_title=column,
                yaxis_title='Count',
                bargap=0.05,
                barmode='overlay',
                height=400
            )
            
            # Create box plot comparison
            fig_box = go.Figure()
            
            fig_box.add_trace(go.Box(
                y=original_df[column].dropna(),
                name='Before',
                marker_color='#3498db'
            ))
            
            fig_box.add_trace(go.Box(
                y=self.df[column].dropna(),
                name='After',
                marker_color='#e74c3c'
            ))
            
            fig_box.update_layout(
                title_text=f'Box Plot of {column} Before and After Cleaning',
                yaxis_title=column,
                height=400
            )
            
            # Calculate basic statistics
            stats = {
                'before': {
                    'mean': float(original_df[column].mean()),
                    'median': float(original_df[column].median()),
                    'std': float(original_df[column].std()),
                    'min': float(original_df[column].min()),
                    'max': float(original_df[column].max()),
                    'missing': int(original_df[column].isna().sum())
                },
                'after': {
                    'mean': float(self.df[column].mean()),
                    'median': float(self.df[column].median()),
                    'std': float(self.df[column].std()),
                    'min': float(self.df[column].min()),
                    'max': float(self.df[column].max()),
                    'missing': int(self.df[column].isna().sum())
                }
            }
            
            return {
                'column_type': col_type,
                'plots': {
                    'histogram_overlay': fig,
                    'box_comparison': fig_box
                },
                'stats': stats
            }
            
        elif col_type in ['categorical', 'boolean']:
            # Get value counts
            before_counts = original_df[column].value_counts().reset_index()
            before_counts.columns = [column, 'Before']
            
            after_counts = self.df[column].value_counts().reset_index()
            after_counts.columns = [column, 'After']
            
            # Merge the counts
            merged_counts = pd.merge(before_counts, after_counts, on=column, how='outer')
            merged_counts = merged_counts.fillna(0)
            
            # Calculate the difference
            merged_counts['Difference'] = merged_counts['After'] - merged_counts['Before']
            
            # Create bar chart for comparison
            fig_bar = go.Figure()
            
            fig_bar.add_trace(go.Bar(
                x=merged_counts[column],
                y=merged_counts['Before'],
                name='Before',
                marker_color='#3498db'
            ))
            
            fig_bar.add_trace(go.Bar(
                x=merged_counts[column],
                y=merged_counts['After'],
                name='After',
                marker_color='#e74c3c'
            ))
            
            fig_bar.update_layout(
                title_text=f'Distribution of {column} Before and After Cleaning',
                xaxis_title=column,
                yaxis_title='Count',
                barmode='group',
                height=500,
                xaxis_tickangle=-45
            )
            
            # Create a horizontal bar chart for changes
            top_changes = merged_counts.copy()
            top_changes['Abs_Diff'] = abs(top_changes['Difference'])
            top_changes = top_changes.sort_values('Abs_Diff', ascending=False).head(10)
            
            fig_diff = px.bar(
                top_changes,
                y=column,
                x='Difference',
                title=f'Top Changes in {column} After Cleaning',
                color='Difference',
                color_continuous_scale=['#e74c3c', '#f1c40f', '#2ecc71'],
                height=400
            )
            
            # Calculate basic statistics
            stats = {
                'before': {
                    'unique_values': int(original_df[column].nunique()),
                    'top_value': str(original_df[column].mode()[0]) if not original_df[column].mode().empty else None,
                    'missing': int(original_df[column].isna().sum()),
                    'value_counts': before_counts.set_index(column)['Before'].to_dict()
                },
                'after': {
                    'unique_values': int(self.df[column].nunique()),
                    'top_value': str(self.df[column].mode()[0]) if not self.df[column].mode().empty else None,
                    'missing': int(self.df[column].isna().sum()),
                    'value_counts': after_counts.set_index(column)['After'].to_dict()
                }
            }
            
            return {
                'column_type': col_type,
                'plots': {
                    'bar_comparison': fig_bar,
                    'changes': fig_diff
                },
                'stats': stats
            }
        
        else:
            # For other types, just compare missing values
            before_missing = original_df[column].isna().sum()
            after_missing = self.df[column].isna().sum()
            
            missing_data = pd.DataFrame({
                'Dataset': ['Before', 'After'],
                'Missing Count': [before_missing, after_missing],
                'Missing Percentage': [before_missing / len(original_df) * 100, 
                                    after_missing / len(self.df) * 100]
            })
            
            fig_missing = px.bar(
                missing_data,
                x='Dataset',
                y='Missing Percentage',
                title=f'Missing Values in {column} Before and After Cleaning',
                color='Dataset',
                color_discrete_map={'Before': '#3498db', 'After': '#e74c3c'},
                text='Missing Count',
                height=400
            )
            
            return {
                'column_type': 'other',
                'plots': {
                    'missing_comparison': fig_missing
                },
                'stats': {
                    'before_missing': int(before_missing),
                    'after_missing': int(after_missing)
                }
            }
    
    def create_correlation_heatmap(self) -> Optional[go.Figure]:
        """
        Create a correlation heatmap for numerical variables.
        
        Returns:
            Plotly figure with correlation heatmap
        """
        # Check if there are enough numerical columns
        if len(self.numeric_cols) < 2:
            return None
            
        # Calculate correlation matrix
        corr_matrix = self.df[self.numeric_cols].corr()
        
        # Create heatmap
        fig = px.imshow(
            corr_matrix,
            labels=dict(x="Column", y="Column", color="Correlation"),
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            color_continuous_scale=['#e74c3c', '#ffffff', '#2ecc71'],
            zmin=-1, zmax=1,
            title='Correlation Heatmap of Numerical Variables',
            height=600
        )
        
        # Add correlation values as text
        text_annotations = []
        for i, row in enumerate(corr_matrix.values):
            for j, value in enumerate(row):
                text_annotations.append(
                    dict(
                        x=j, y=i,
                        text=f"{value:.2f}",
                        font=dict(color='black' if abs(value) < 0.7 else 'white'),
                        showarrow=False
                    )
                )
                
        fig.update_layout(annotations=text_annotations)
        
        return fig
    
    def create_data_quality_dashboard(self) -> Dict[str, Any]:
        """
        Create a comprehensive dashboard for data quality assessment.
            
        Returns:
            Dictionary with multiple visualizations for data quality
        """
        dashboard = {}
        
        # 1. Missing values overview
        missing_data = self.plot_missing_values()
        dashboard['missing_values'] = missing_data
        
        # 2. Correlation heatmap
        dashboard['correlation_heatmap'] = self.create_correlation_heatmap()
        
        # 3. Column type distribution
        column_types = pd.DataFrame({
            'Type': list(self.column_types.values())
        })
        type_counts = column_types['Type'].value_counts().reset_index()
        type_counts.columns = ['Column Type', 'Count']
        
        fig_types = px.pie(
            type_counts,
            names='Column Type',
            values='Count',
            title='Distribution of Column Types',
            height=400
        )
        dashboard['column_types'] = {
            'plot': fig_types,
            'counts': type_counts.to_dict(orient='records')
        }
        
        # 4. Data quality score
        # Simple scoring based on missing values and column types
        total_cells = self.df.shape[0] * self.df.shape[1]
        missing_cells = self.df.isna().sum().sum()
        missing_score = 100 - (missing_cells / total_cells * 100)
        
        # Better score if we have more numerical columns (for analytics)
        num_score = len(self.numeric_cols) / len(self.df.columns) * 100
        
        # Final quality score (weighted average)
        quality_score = missing_score * 0.7 + num_score * 0.3
        
        dashboard['quality_score'] = {
            'overall': round(quality_score, 2),
            'completeness': round(missing_score, 2),
            'numeric_ratio': round(num_score, 2)
        }
        
        return dashboard
