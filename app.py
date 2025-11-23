# comprehensive_data_analyzer.py (corrected)
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO
import base64
import warnings
import pandas.api.types as ptypes

warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="AI Data Analyzer Pro",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.8rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
        background: linear-gradient(45deg, #1f77b4, #2ca02c);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 15px;
        border-left: 4px solid #1f77b4;
        margin: 10px 0;
    }
    .chart-header {
        font-size: 1.3rem;
        color: #2e86ab;
        margin-bottom: 1rem;
        border-bottom: 2px solid #e9ecef;
        padding-bottom: 0.5rem;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 5px;
        padding: 15px;
        margin: 10px 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 5px;
        padding: 15px;
        margin: 10px 0;
    }
    .info-box {
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        border-radius: 5px;
        padding: 15px;
        margin: 10px 0;
    }
    .tab-container {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)


class SmartDashboardBuilder:
    def __init__(self, df):
        self.df = df.copy()
        # Detect numeric, categorical, datetime robustly
        self.numeric_cols = [c for c in df.columns if ptypes.is_numeric_dtype(df[c])]
        self.categorical_cols = [c for c in df.columns if (ptypes.is_object_dtype(df[c]) or ptypes.is_categorical_dtype(df[c]))]
        self.datetime_cols = [c for c in df.columns if ptypes.is_datetime64_any_dtype(df[c])]

    def analyze_column(self, column):
        """Analyze a column to determine the best chart type"""
        col_data = self.df[column]

        # Basic stats
        analysis = {
            'type': 'unknown',
            'unique_count': int(col_data.nunique(dropna=True)),
            'null_count': int(col_data.isnull().sum()),
            'suggested_charts': []
        }

        # Numeric columns
        if column in self.numeric_cols:
            analysis['type'] = 'numeric'
            analysis['min'] = col_data.min()
            analysis['max'] = col_data.max()
            analysis['mean'] = col_data.mean()
            analysis['std'] = col_data.std()

            # Chart suggestions for numeric
            if col_data.nunique() <= 10:
                analysis['suggested_charts'].extend(['histogram', 'box_plot', 'violin_plot'])
            else:
                analysis['suggested_charts'].extend(['histogram', 'density_plot', 'box_plot'])

        # Categorical columns
        elif column in self.categorical_cols:
            analysis['type'] = 'categorical'
            analysis['value_counts'] = col_data.value_counts()

            # Chart suggestions for categorical
            if col_data.nunique() <= 8:
                analysis['suggested_charts'].extend(['bar_chart', 'pie_chart', 'donut_chart'])
            elif col_data.nunique() <= 20:
                analysis['suggested_charts'].extend(['bar_chart', 'treemap'])
            else:
                analysis['suggested_charts'].extend(['bar_chart', 'horizontal_bar'])

        # Datetime columns
        elif column in self.datetime_cols:
            analysis['type'] = 'datetime'
            analysis['suggested_charts'].extend(['line_chart', 'area_chart', 'calendar_heatmap'])

        return analysis

    def detect_relationships(self):
        """Detect relationships between columns for multi-variable charts"""
        relationships = []

        # Numeric vs Numeric
        if len(self.numeric_cols) >= 2:
            relationships.append({
                'type': 'scatter',
                'columns': self.numeric_cols[:2],
                'description': f'Relationship between {self.numeric_cols[0]} and {self.numeric_cols[1]}'
            })

        # Categorical vs Numeric
        if self.categorical_cols and self.numeric_cols:
            relationships.append({
                'type': 'box',
                'columns': [self.categorical_cols[0], self.numeric_cols[0]],
                'description': f'{self.numeric_cols[0]} distribution across {self.categorical_cols[0]}'
            })

        # Time series if datetime exists
        if self.datetime_cols and self.numeric_cols:
            relationships.append({
                'type': 'time_series',
                'columns': [self.datetime_cols[0], self.numeric_cols[0]],
                'description': f'{self.numeric_cols[0]} over time'
            })

        return relationships

    def create_histogram(self, column, title=None):
        """Create histogram for numeric column"""
        fig = px.histogram(
            self.df, x=column,
            title=title or f'Distribution of {column}',
            nbins=30,
            color_discrete_sequence=['#1f77b4']
        )
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        return fig

    def create_bar_chart(self, column, title=None):
        """Create bar chart for categorical column"""
        value_counts = self.df[column].value_counts().head(15)  # Limit to top 15
        fig = px.bar(
            x=value_counts.index, y=value_counts.values,
            title=title or f'Top values in {column}',
            labels={'x': column, 'y': 'Count'},
            color=value_counts.values,
            color_continuous_scale='Blues'
        )
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        return fig

    def create_pie_chart(self, column, title=None):
        """Create pie chart for categorical column"""
        value_counts = self.df[column].value_counts().head(8)  # Limit to top 8
        fig = px.pie(
            values=value_counts.values, names=value_counts.index,
            title=title or f'Distribution of {column}',
            color_discrete_sequence=px.colors.sequential.Blues_r
        )
        return fig

    def create_box_plot(self, column, title=None):
        """Create box plot for numeric column"""
        fig = px.box(
            self.df, y=column,
            title=title or f'Box plot of {column}',
            color_discrete_sequence=['#2ca02c']
        )
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        return fig

    def create_scatter_plot(self, x_col, y_col, title=None):
        """Create scatter plot for two numeric columns"""
        fig = px.scatter(
            self.df, x=x_col, y=y_col,
            title=title or f'{x_col} vs {y_col}',
            trendline='lowess',
            color_discrete_sequence=['#ff7f0e']
        )
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        return fig

    def create_line_chart(self, x_col, y_col, title=None):
        """Create line chart for time series"""
        fig = px.line(
            self.df, x=x_col, y=y_col,
            title=title or f'{y_col} over time',
            color_discrete_sequence=['#d62728']
        )
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        return fig

    def create_heatmap(self, title="Correlation Heatmap"):
        """Create correlation heatmap for numeric columns"""
        if len(self.numeric_cols) < 2:
            return None

        corr_matrix = self.df[self.numeric_cols].corr()
        fig = px.imshow(
            corr_matrix,
            title=title,
            color_continuous_scale='RdBu_r',
            aspect="auto"
        )
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        return fig

    def generate_dashboard(self):
        """Generate complete dashboard with AI-selected charts"""
        dashboard = {
            'single_column_charts': [],
            'relationship_charts': [],
            'summary_metrics': []
        }

        # Analyze each column and create appropriate charts
        for column in self.df.columns:
            analysis = self.analyze_column(column)

            if analysis['type'] == 'numeric':
                # For numeric columns, create histogram and box plot
                dashboard['single_column_charts'].append({
                    'type': 'histogram',
                    'column': column,
                    'chart': self.create_histogram(column),
                    'description': f'Distribution of {column}'
                })

                if analysis['unique_count'] > 5:  # Only box plot if enough variation
                    dashboard['single_column_charts'].append({
                        'type': 'box_plot',
                        'column': column,
                        'chart': self.create_box_plot(column),
                        'description': f'Statistical summary of {column}'
                    })

            elif analysis['type'] == 'categorical':
                # For categorical, create bar chart
                dashboard['single_column_charts'].append({
                    'type': 'bar_chart',
                    'column': column,
                    'chart': self.create_bar_chart(column),
                    'description': f'Frequency of {column} values'
                })

                # Add pie chart only for columns with few unique values
                if analysis['unique_count'] <= 8:
                    dashboard['single_column_charts'].append({
                        'type': 'pie_chart',
                        'column': column,
                        'chart': self.create_pie_chart(column),
                        'description': f'Proportion of {column} values'
                    })

        # Create relationship charts
        relationships = self.detect_relationships()
        for rel in relationships:
            if rel['type'] == 'scatter' and len(rel['columns']) == 2:
                dashboard['relationship_charts'].append({
                    'type': 'scatter',
                    'columns': rel['columns'],
                    'chart': self.create_scatter_plot(rel['columns'][0], rel['columns'][1]),
                    'description': rel['description']
                })
            elif rel['type'] == 'time_series' and len(rel['columns']) == 2:
                dashboard['relationship_charts'].append({
                    'type': 'line_chart',
                    'columns': rel['columns'],
                    'chart': self.create_line_chart(rel['columns'][0], rel['columns'][1]),
                    'description': rel['description']
                })

        # Add correlation heatmap if we have multiple numeric columns
        heatmap = self.create_heatmap()
        if heatmap:
            dashboard['relationship_charts'].append({
                'type': 'heatmap',
                'columns': self.numeric_cols,
                'chart': heatmap,
                'description': 'Correlation between numeric variables'
            })

        return dashboard


class DataCleaner:
    def __init__(self, df):
        self.df = df.copy()
        self.original_df = df.copy()
        self.cleaning_log = []

    def detect_missing_values(self):
        """Detect missing values in the dataset"""
        missing_data = self.df.isnull().sum()
        missing_percent = (missing_data / len(self.df)) * 100
        missing_info = pd.DataFrame({
            'Column': missing_data.index,
            'Missing_Count': missing_data.values,
            'Missing_Percent': missing_percent.values
        })
        return missing_info[missing_info['Missing_Count'] > 0]

    def handle_missing_values(self, strategy='drop', fill_value=None, specific_columns=None):
        """Handle missing values based on selected strategy"""
        if strategy == 'drop':
            original_rows = len(self.df)
            self.df = self.df.dropna()
            removed_rows = original_rows - len(self.df)
            self.cleaning_log.append(f"Dropped rows with missing values. Removed {removed_rows} rows")
        elif strategy == 'fill':
            # fill_value expected: 'mean', 'median', or 'mode'
            cols_to_fill = specific_columns if specific_columns else self.df.columns
            for col in cols_to_fill:
                if self.df[col].isnull().any():
                    if ptypes.is_numeric_dtype(self.df[col]):
                        if fill_value == 'mean':
                            fill_val = self.df[col].mean()
                        elif fill_value == 'median':
                            fill_val = self.df[col].median()
                        else:
                            # default to median if unspecified
                            fill_val = self.df[col].median()
                    else:
                        # categorical -> use mode if available
                        modes = self.df[col].mode()
                        fill_val = modes[0] if len(modes) > 0 else 'Unknown'
                    self.df[col].fillna(fill_val, inplace=True)
                    self.cleaning_log.append(f"Filled missing values in '{col}' with '{fill_val}'")

    def detect_duplicates(self):
        """Detect duplicate rows"""
        return int(self.df.duplicated().sum())

    def remove_duplicates(self):
        """Remove duplicate rows"""
        original_count = len(self.df)
        self.df = self.df.drop_duplicates()
        removed_count = original_count - len(self.df)
        if removed_count > 0:
            self.cleaning_log.append(f"Removed {removed_count} duplicate rows")
        return removed_count

    def detect_outliers_iqr(self, column):
        """Detect outliers using IQR method"""
        if ptypes.is_numeric_dtype(self.df[column]):
            Q1 = self.df[column].quantile(0.25)
            Q3 = self.df[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = self.df[(self.df[column] < lower_bound) | (self.df[column] > upper_bound)]
            return int(len(outliers)), lower_bound, upper_bound
        return 0, None, None

    def handle_outliers(self, column, method='cap', custom_bounds=None):
        """Handle outliers in numerical columns"""
        if not ptypes.is_numeric_dtype(self.df[column]):
            return

        Q1 = self.df[column].quantile(0.25)
        Q3 = self.df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        if method == 'cap':
            self.df[column] = np.where(self.df[column] < lower_bound, lower_bound,
                                      np.where(self.df[column] > upper_bound, upper_bound, self.df[column]))
            self.cleaning_log.append(f"Capped outliers in '{column}' using IQR method")
        elif method == 'remove':
            original_len = len(self.df)
            self.df = self.df[(self.df[column] >= lower_bound) & (self.df[column] <= upper_bound)]
            removed = original_len - len(self.df)
            self.cleaning_log.append(f"Removed {removed} rows with outliers in '{column}'")

    def generate_ai_suggestions(self):
        """Generate AI-powered cleaning suggestions"""
        suggestions = []

        # Analyze missing values
        missing_info = self.detect_missing_values()
        if not missing_info.empty:
            for _, row in missing_info.iterrows():
                col = row['Column']
                percent = row['Missing_Percent']
                if percent > 50:
                    suggestions.append(f"🚨 Column '{col}' has {percent:.1f}% missing values - consider dropping this column")
                elif percent > 20:
                    suggestions.append(f"⚠️ Column '{col}' has {percent:.1f}% missing values - consider imputation or investigation")
                else:
                    if ptypes.is_numeric_dtype(self.df[col]):
                        suggestions.append(f"💡 Column '{col}' has {percent:.1f}% missing values - fill with median")
                    else:
                        suggestions.append(f"💡 Column '{col}' has {percent:.1f}% missing values - fill with mode")

        # Analyze outliers in numerical columns
        numerical_cols = [c for c in self.df.columns if ptypes.is_numeric_dtype(self.df[c])]
        for col in numerical_cols:
            outlier_count, lower, upper = self.detect_outliers_iqr(col)
            if outlier_count > 0:
                percent = (outlier_count / len(self.df)) * 100
                if percent > 10:
                    suggestions.append(f"🚨 Column '{col}' has {outlier_count} outliers ({percent:.1f}%) - consider capping or removal")
                else:
                    suggestions.append(f"💡 Column '{col}' has {outlier_count} outliers - consider capping at bounds [{lower:.2f}, {upper:.2f}]")

        # Check for duplicates
        duplicate_count = self.detect_duplicates()
        if duplicate_count > 0:
            suggestions.append(f"🔍 Found {duplicate_count} duplicate rows - recommend removal")

        # Data type suggestions
        for col in self.df.columns:
            if ptypes.is_object_dtype(self.df[col]):
                unique_ratio = self.df[col].nunique(dropna=True) / max(1, len(self.df))
                if unique_ratio < 0.1:
                    suggestions.append(f"🏷️ Column '{col}' has low cardinality ({self.df[col].nunique()} unique values) - consider converting to category")

        return suggestions

    def get_summary_stats(self):
        """Get summary statistics for before/after comparison"""
        return {
            'shape': self.df.shape,
            'memory_usage': int(self.df.memory_usage(deep=True).sum()),
            'total_missing': int(self.df.isnull().sum().sum()),
            'total_duplicates': int(self.df.duplicated().sum()),
            'numerical_columns': len([c for c in self.df.columns if ptypes.is_numeric_dtype(self.df[c])]),
            'categorical_columns': len([c for c in self.df.columns if ptypes.is_object_dtype(self.df[c])])
        }


def create_download_link(df, filename="cleaned_data.csv"):
    """Create a download link for the cleaned dataframe"""
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">📥 Download Cleaned CSV</a>'
    return href


def try_parse_dates(df):
    """Try to convert any column with 'date' in its name to datetime"""
    for col in df.columns:
        if 'date' in col.lower():
            try:
                df[col] = pd.to_datetime(df[col], errors='coerce')
            except Exception:
                pass
    return df


def main():
    st.markdown('<h1 class="main-header">🔍 AI Data Analyzer Pro</h1>', unsafe_allow_html=True)
    st.markdown("### Your All-in-One Solution for Data Cleaning & Visualization")

    # Sidebar
    with st.sidebar:
        st.header("📁 Data Upload")

        if 'sample_data' in st.session_state:
            uploaded_file = None
            df = pd.read_csv(StringIO(st.session_state.sample_data))
            st.success("✅ Sample data loaded!")
        else:
            uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
            df = None

        st.header("🎯 Sample Datasets")
        col1, col2 = st.columns(2)

        with col1:
            if st.button("📈 Sales Data"):
                sales_data = """Date,Product,Category,Region,Sales,Quantity,Profit,Customer_Rating
2024-01-01,Laptop,Electronics,North,1200,5,240,4.2
2024-01-02,Smartphone,Electronics,South,800,8,160,4.5
2024-01-03,Desk,Furniture,East,300,3,60,3.8
2024-01-04,Chair,Furniture,West,150,10,30,4.1
2024-01-05,Monitor,Electronics,North,250,4,75,4.3
2024-01-06,Keyboard,Electronics,South,80,12,16,4.0
2024-01-07,Notebook,Office,East,5,50,1,3.5
2024-01-08,Tablet,Electronics,West,600,2,120,4.7
2024-01-09,Printer,Electronics,North,350,3,70,4.1
2024-01-10,Scanner,Electronics,South,200,5,40,3.9
2024-01-11,Laptop,Electronics,East,1200,4,240,4.4
2024-01-12,Smartphone,Electronics,West,800,6,160,4.6
2024-01-13,Desk,Furniture,North,300,2,60,3.9
2024-01-14,Chair,Furniture,South,150,8,30,4.2
2024-01-15,Monitor,Electronics,East,250,5,75,4.3"""
                st.session_state.sample_data = sales_data
                st.rerun()


        with col2:
            if st.button("👥 Customer Data"):
                customer_data = """CustomerID,Age,Gender,City,Salary,CreditScore,SpendingScore,Membership
C001,28,Male,New York,50000,750,85,Gold
C002,34,Female,California,75000,800,92,Platinum
C003,45,Male,Texas,60000,650,78,Silver
C004,29,Female,Florida,55000,720,88,Gold
C005,52,Male,Illinois,90000,850,95,Platinum
C006,31,Female,Washington,48000,680,76,Silver
C007,41,Male,Georgia,82000,790,89,Gold
C008,26,Female,Michigan,45000,710,82,Silver
C009,38,Male,Ohio,67000,770,91,Platinum
C010,33,Female,Arizona,58000,730,84,Gold
C011,47,Male,Colorado,88000,820,94,Platinum
C012,30,Female,Tennessee,52000,690,79,Silver
C013,35,Male,Missouri,63000,740,86,Gold
C014,42,Female,Virginia,78000,810,93,Platinum
C015,27,Male,Maryland,47000,670,77,Silver"""
                st.session_state.sample_data = customer_data
                st.rerun()

        if st.button("🔄 Clear Data"):
            if 'sample_data' in st.session_state:
                del st.session_state.sample_data
            st.rerun()

    # Main content
    if 'sample_data' in st.session_state or uploaded_file is not None:
        try:
            # Load data
            if 'sample_data' in st.session_state:
                df = pd.read_csv(StringIO(st.session_state.sample_data))
            else:
                df = pd.read_csv(uploaded_file)

            # Try to parse date-like columns
            df = try_parse_dates(df)

            # Initialize dashboard builder (move before tabs so available in all tabs)
            builder = SmartDashboardBuilder(df)

            # Create main tabs for different functionalities
            tab1, tab2, tab3 = st.tabs(["📊 Smart Dashboard", "🧹 Data Cleaning", "🔍 Data Overview"])

            with tab1:
                st.markdown('<h2 class="chart-header">🤖 AI-Powered Dashboard Builder</h2>', unsafe_allow_html=True)

                dashboard = builder.generate_dashboard()

                # Display dataset info
                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.metric("Total Rows", len(df))
                with col2:
                    st.metric("Total Columns", len(df.columns))
                with col3:
                    st.metric("Numeric Columns", len(builder.numeric_cols))
                with col4:
                    st.metric("Categorical Columns", len(builder.categorical_cols))

                # Dashboard layout
                st.markdown("---")

                # Single column charts
                if dashboard['single_column_charts']:
                    st.subheader("📈 Single Variable Analysis")

                    # Create 2-column layout for charts
                    cols = st.columns(2)
                    for i, chart_info in enumerate(dashboard['single_column_charts']):
                        with cols[i % 2]:
                            st.plotly_chart(chart_info['chart'], use_container_width=True)
                            st.caption(f"**{chart_info['description']}**")

                # Relationship charts
                if dashboard['relationship_charts']:
                    st.subheader("🔗 Multi-Variable Analysis")

                    cols = st.columns(2)
                    for i, chart_info in enumerate(dashboard['relationship_charts']):
                        with cols[i % 2]:
                            st.plotly_chart(chart_info['chart'], use_container_width=True)
                            st.caption(f"**{chart_info['description']}**")

                # Data insights
                with st.expander("💡 AI Insights & Recommendations"):
                    insights = []

                    # Numeric columns insights
                    for col in builder.numeric_cols:
                        analysis = builder.analyze_column(col)
                        # guard against None/NaN
                        mean = analysis.get('mean') or 0
                        std = analysis.get('std') or 0
                        if std and mean and std > abs(mean) * 0.5:
                            insights.append(f"📊 **{col}** has high variability (std: {analysis['std']:.2f})")
                        if analysis['null_count'] > 0:
                            insights.append(f"⚠️ **{col}** has {analysis['null_count']} missing values")

                    # Categorical columns insights
                    for col in builder.categorical_cols:
                        analysis = builder.analyze_column(col)
                        if analysis['unique_count'] > 20:
                            insights.append(f"🎯 **{col}** has high cardinality ({analysis['unique_count']} unique values)")

                    if insights:
                        for insight in insights:
                            st.write(f"• {insight}")
                    else:
                        st.success("✅ Your data looks clean and well-structured!")

                    # Chart recommendations
                    st.write("### 📈 Recommended Additional Analysis:")
                    if len(builder.numeric_cols) >= 3:
                        st.write("• Consider creating a pair plot for numeric variable relationships")
                    if builder.datetime_cols and len(builder.numeric_cols) >= 2:
                        st.write("• Try a multi-line chart to compare trends over time")
                    if len(builder.categorical_cols) >= 2:
                        st.write("• Create a stacked bar chart to analyze category combinations")

            with tab2:
                st.markdown('<h2 class="chart-header">🧹 AI-Powered Data Cleaning</h2>', unsafe_allow_html=True)

                cleaner = DataCleaner(df)

                # Cleaning options sidebar within the tab
                col1, col2 = st.columns([1, 2])

                with col1:
                    st.subheader("⚙️ Cleaning Options")

                    st.write("**Missing Values**")
                    missing_strategy = st.selectbox(
                        "Handle missing values:",
                        ["None", "Drop rows", "Fill with mean", "Fill with median", "Fill with mode"]
                    )

                    st.write("**Duplicates**")
                    remove_duplicates = st.checkbox("Remove duplicate rows", value=True)

                    st.write("**Outliers**")
                    outlier_handling = st.selectbox(
                        "Handle outliers:",
                        ["None", "Cap using IQR", "Remove outliers"]
                    )

                    if st.button("🚀 Apply Automated Cleaning", type="primary"):
                        # Store original state
                        original_shape = cleaner.df.shape

                        # Apply cleaning based on selections
                        if missing_strategy == "Drop rows":
                            cleaner.handle_missing_values(strategy='drop')
                        elif missing_strategy == "Fill with mean":
                            cleaner.handle_missing_values(strategy='fill', fill_value='mean')
                        elif missing_strategy == "Fill with median":
                            cleaner.handle_missing_values(strategy='fill', fill_value='median')
                        elif missing_strategy == "Fill with mode":
                            cleaner.handle_missing_values(strategy='fill', fill_value='mode')

                        if remove_duplicates:
                            cleaner.remove_duplicates()

                        if outlier_handling == "Cap using IQR":
                            numerical_cols = [c for c in cleaner.df.columns if ptypes.is_numeric_dtype(cleaner.df[c])]
                            for col in numerical_cols:
                                cleaner.handle_outliers(col, method='cap')
                        elif outlier_handling == "Remove outliers":
                            numerical_cols = [c for c in cleaner.df.columns if ptypes.is_numeric_dtype(cleaner.df[c])]
                            for col in numerical_cols:
                                cleaner.handle_outliers(col, method='remove')

                        st.success(f"✅ Cleaning completed! Dataset shape changed from {original_shape} to {cleaner.df.shape}")

                with col2:
                    # Data Quality Report
                    st.subheader("🔍 Data Quality Report")

                    # Missing values analysis
                    missing_info = cleaner.detect_missing_values()
                    if not missing_info.empty:
                        st.write("**❌ Missing Values**")
                        st.dataframe(missing_info)
                    else:
                        st.markdown('<div class="success-box">✅ No missing values found!</div>', unsafe_allow_html=True)

                    # Duplicates analysis
                    duplicate_count = cleaner.detect_duplicates()
                    st.write("**🔍 Duplicate Rows**")
                    if duplicate_count > 0:
                        st.markdown(f'<div class="warning-box">⚠️ Found {duplicate_count} duplicate rows</div>', unsafe_allow_html=True)
                    else:
                        st.markdown('<div class="success-box">✅ No duplicate rows found!</div>', unsafe_allow_html=True)

                    # AI Suggestions
                    st.subheader("🤖 AI Cleaning Suggestions")
                    suggestions = cleaner.generate_ai_suggestions()
                    if suggestions:
                        for i, suggestion in enumerate(suggestions[:10], 1):  # Show top suggestions
                            st.write(f"{i}. {suggestion}")
                    else:
                        st.markdown('<div class="success-box">🎉 Your data looks clean! No major issues detected.</div>', unsafe_allow_html=True)

                    # Download cleaned data
                    st.markdown("---")
                    st.markdown("### 📥 Download Cleaned Data")
                    st.markdown(create_download_link(cleaner.df), unsafe_allow_html=True)

            with tab3:
                st.markdown('<h2 class="chart-header">🔍 Comprehensive Data Overview</h2>', unsafe_allow_html=True)

                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Dataset Shape:**", df.shape)
                    st.write("**Columns:**", list(df.columns))

                with col2:
                    st.write("**Data Types:**")
                    st.write(df.dtypes)

                st.subheader("First 10 Rows")
                st.dataframe(df.head(10))

                st.subheader("Basic Statistics")
                st.write(df.describe(include='all').T)

                # Column analysis
                with st.expander("📋 Detailed Column Analysis"):
                    for col in df.columns:
                        analysis = builder.analyze_column(col)
                        col1, col2 = st.columns([1, 3])
                        with col1:
                            st.write(f"**{col}**")
                            st.write(f"Type: {analysis['type']}")
                        with col2:
                            st.write(f"Suggested charts: {', '.join(analysis['suggested_charts'])}")

        except Exception as e:
            st.error(f"Error processing data: {str(e)}")

    else:
        # Welcome section
        st.markdown("""
        <div style='background-color: #f8f9fa; padding: 2rem; border-radius: 10px; border-left: 4px solid #1f77b4;'>
        <h3>🚀 Welcome to AI Data Analyzer Pro!</h3>
        <p>Your all-in-one solution for data analysis, cleaning, and visualization.</p>

        <h4>🎯 Key Features:</h4>

        <h5>📊 Smart Dashboard Builder:</h5>
        <ul>
            <li>Automatic chart type selection based on data characteristics</li>
            <li>Interactive Plotly visualizations</li>
            <li>AI-powered insights and recommendations</li>
            <li>Professional dashboard layout</li>
        </ul>

        <h5>🧹 Data Cleaning Tool:</h5>
        <ul>
            <li>Automatic detection of missing values, duplicates, and outliers</li>
            <li>AI-powered cleaning suggestions</li>
            <li>One-click automated cleaning</li>
            <li>Before/After comparison</li>
        </ul>

        <h5>🔍 Data Overview:</h5>
        <ul>
            <li>Comprehensive data profiling</li>
            <li>Statistical summaries</li>
            <li>Column-wise analysis</li>
            <li>Data quality assessment</li>
        </ul>

        <p><strong>Get started:</strong> Upload a CSV file or try one of our sample datasets!</p>
        </div>
        """, unsafe_allow_html=True)

        # Sample data options
        st.markdown("### 🧪 Try with Sample Data")
        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button("📊 Employee Data", use_container_width=True):
                employee_data = """ID,Name,Age,Salary,Department,Join_Date,Experience,City,Performance_Score
1,John Smith,28,50000,IT,2020-01-15,4,New York,85
2,Jane Doe,32,60000,HR,2019-03-20,5,Chicago,92
3,Bob Johnson,45,75000,Finance,2015-06-10,9,New York,78
4,Alice Brown,29,55000,IT,2021-02-28,3,Chicago,88
5,Charlie Wilson,35,65000,Marketing,2018-11-05,6,Los Angeles,76
6,Diana Lee,41,80000,Finance,2016-09-12,8,New York,95
7,John Smith,28,50000,IT,2020-01-15,4,New York,85
8,Mike Davis,26,48000,Sales,2022-01-10,2,Chicago,82
9,Sarah Miller,38,70000,HR,2017-04-22,7,Los Angeles,90
10,Kevin Taylor,52,90000,Finance,2010-08-30,14,New York,87
11,Lisa Anderson,29,,Marketing,2021-07-14,3,Chicago,79
12,Tom Wilson,61,120000,IT,2005-12-01,19,New York,91
13,Emma Garcia,27,52000,Sales,2022-03-18,2,Los Angeles,84
14,Robert Brown,34,62000,Marketing,2019-09-25,5,Chicago,
15,Maria Martinez,31,58000,HR,2020-11-08,4,New York,86
16,Outlier Test,150,1000000,IT,2020-01-01,5,New York,50"""
                st.session_state.sample_data = employee_data
                st.rerun()

        with col2:
            if st.button("🏥 Healthcare Data", use_container_width=True):
                healthcare_data = """PatientID,Name,Age,Gender,BloodPressure,Cholesterol,HeartRate,Temperature,BloodSugar,BMI,Smoker,Diabetes_Risk
P001,John Smith,45,Male,120/80,180,72,98.6,95,24.5,No,Low
P002,Maria Garcia,52,Female,130/85,200,75,98.4,110,26.8,Yes,Medium
P003,Robert Johnson,38,Male,118/78,160,68,98.7,92,23.1,No,Low
P004,Lisa Brown,61,Female,140/90,240,80,98.2,145,29.3,Yes,High
P005,Michael Davis,29,Male,122/79,170,70,98.5,88,22.4,No,Low
P006,Sarah Wilson,47,Female,125/82,190,73,98.3,105,25.6,No,Medium
P007,David Miller,55,Male,135/88,220,78,98.1,135,28.2,Yes,High
P008,Jennifer Taylor,42,Female,128/84,185,74,98.6,98,24.9,No,Low
P009,Outlier Patient,150,Male,,500,200,95.0,300,45.0,Yes,High
P010,Missing Data,,Male,120/80,180,72,98.6,95,24.5,,"""
                st.session_state.sample_data = healthcare_data
                st.rerun()

        with col3:
            if st.button("🏠 Housing Data", use_container_width=True):
                housing_data = """Price,SquareFeet,Bedrooms,Bathrooms,YearBuilt,Location,Pool,Garage,Rating
350000,1800,3,2,1995,Suburb,No,Yes,4.2
450000,2200,4,2.5,2005,Urban,Yes,Yes,4.5
275000,1500,3,1,1985,Rural,No,No,3.8
520000,2400,4,3,2010,Suburb,Yes,Yes,4.7
310000,1600,3,2,2000,Urban,No,Yes,4.1
280000,1550,2,1,1990,Rural,No,No,3.9
475000,2100,3,2.5,2008,Suburb,Yes,Yes,4.6
325000,1700,3,2,1998,Urban,No,Yes,4.3
295000,1450,2,1,1988,Rural,No,No,3.7
500000,2300,4,3,2012,Suburb,Yes,Yes,4.8
10000000,10000,10,8,2020,Urban,Yes,Yes,5.0
,,3,2,2000,Suburb,No,Yes,4.0"""
                st.session_state.sample_data = housing_data
                st.rerun() 


if __name__ == "__main__":
    main()
