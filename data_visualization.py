import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


class DataVisualizer:
    def plot_histogram(self, df: pd.DataFrame, column: str):
        """
        Create a histogram for a numerical column
        
        Args:
            df: The dataframe to visualize
            column: The column to plot
            
        Returns:
            A plotly figure object
        """
        fig = px.histogram(
            df, 
            x=column,
            title=f"Distribution of {column}",
            labels={column: column},
            template="plotly_white",
            color_discrete_sequence=['#3366CC']
        )
        
        fig.update_layout(
            xaxis_title=column,
            yaxis_title="Count",
            bargap=0.1
        )
        
        return fig
    
    def plot_bar_chart(self, df: pd.DataFrame, column: str):
        """
        Create a bar chart for a categorical column
        
        Args:
            df: The dataframe to visualize
            column: The column to plot
            
        Returns:
            A plotly figure object
        """
        # Calculate value counts
        value_counts = df[column].value_counts().reset_index()
        value_counts.columns = [column, 'Count']
        
        fig = px.bar(
            value_counts, 
            x=column, 
            y='Count',
            title=f"Distribution of {column}",
            labels={column: column, 'Count': 'Count'},
            template="plotly_white",
            color_discrete_sequence=['#3366CC']
        )
        
        fig.update_layout(
            xaxis_title=column,
            yaxis_title="Count"
        )
        
        return fig
    
    def plot_box_plot(self, df: pd.DataFrame, column: str):
        """
        Create a box plot for a numerical column
        
        Args:
            df: The dataframe to visualize
            column: The column to plot
            
        Returns:
            A plotly figure object
        """
        fig = px.box(
            df, 
            y=column,
            title=f"Box Plot of {column}",
            labels={column: column},
            template="plotly_white",
            color_discrete_sequence=['#3366CC']
        )
        
        fig.update_layout(
            yaxis_title=column
        )
        
        return fig
    
    def plot_missing_values(self, df: pd.DataFrame):
        """
        Create a bar chart showing missing values by column
        
        Args:
            df: The dataframe to visualize
            
        Returns:
            A plotly figure object
        """
        # Calculate missing values
        missing_values = df.isnull().sum().reset_index()
        missing_values.columns = ['Column', 'Missing Count']
        missing_values = missing_values[missing_values['Missing Count'] > 0]
        
        # Sort by missing count
        missing_values = missing_values.sort_values('Missing Count', ascending=False)
        
        # Calculate missing percentage
        missing_values['Missing Percentage'] = (missing_values['Missing Count'] / len(df) * 100).round(2)
        
        fig = px.bar(
            missing_values, 
            x='Column', 
            y='Missing Count',
            title="Missing Values by Column",
            labels={'Column': 'Column', 'Missing Count': 'Missing Count'},
            template="plotly_white",
            color_discrete_sequence=['#FF6B6B'],
            text='Missing Percentage'
        )
        
        fig.update_layout(
            xaxis_title="Column",
            yaxis_title="Missing Count"
        )
        
        # Add percentage labels
        fig.update_traces(
            texttemplate='%{text:.2f}%',
            textposition='outside'
        )
        
        return fig
    
    def plot_before_after_comparison(self, metric_name: str, before_value: float, after_value: float):
        """
        Create a bar chart comparing before and after values
        
        Args:
            metric_name: The name of the metric being compared
            before_value: The value before cleaning
            after_value: The value after cleaning
            
        Returns:
            A plotly figure object
        """
        fig = go.Figure()
        
        fig.add_trace(
            go.Bar(
                x=['Before', 'After'],
                y=[before_value, after_value],
                text=[f"{before_value:.2f}", f"{after_value:.2f}"],
                textposition='auto',
                marker_color=['#FFB6C1', '#98FB98']
            )
        )
        
        fig.update_layout(
            title=f"{metric_name}: Before vs After",
            xaxis_title="",
            yaxis_title="Value",
            template="plotly_white"
        )
        
        return fig
