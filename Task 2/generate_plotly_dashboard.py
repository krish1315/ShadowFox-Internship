"""
Interactive Plotly Visualizations for Task 2: Store Sales and Profit Analysis
=============================================================================
This script generates an interactive HTML dashboard using Plotly.
It satisfies the requirement for dynamic and interactive visualizations.
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as gg
from plotly.subplots import make_subplots
import os

def create_plotly_dashboard():
    csv_path = 'Sample - Superstore.csv'
    df = pd.read_csv(csv_path, encoding='ISO-8859-1')
    
    # Data Cleaning & DateTime Conversion
    df['Order Date'] = pd.to_datetime(df['Order Date'], format='%m/%d/%Y')
    df['Ship Date'] = pd.to_datetime(df['Ship Date'], format='%m/%d/%Y')
    df['Year'] = df['Order Date'].dt.year
    df['YearMonth'] = df['Order Date'].dt.to_period('M').astype(str)
    
    # Outlier Analysis (IQR Method)
    def detect_iqr_outliers(series):
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        return (series < lower) | (series > upper)
    
    df['Sales_Outlier'] = detect_iqr_outliers(df['Sales'])
    df['Profit_Outlier'] = detect_iqr_outliers(df['Profit'])
    
    # Calculate Sales-to-Profit Ratio
    # Avoid division by zero or negative profit issues by providing clean ratio & margin
    df['Sales_to_Profit_Ratio'] = np.where(df['Profit'] != 0, df['Sales'] / df['Profit'], np.nan)
    df['Profit_Margin_%'] = (df['Profit'] / df['Sales']) * 100
    
    # ---------------------------------------------------------
    # 1. Monthly Temporal Sales & Profit Trends (Interactive Line)
    # ---------------------------------------------------------
    monthly = df.groupby('YearMonth').agg({'Sales': 'sum', 'Profit': 'sum'}).reset_index()

    fig_monthly = make_subplots(specs=[[{"secondary_y": True}]])
    fig_monthly.add_trace(
        gg.Scatter(
            x=monthly['YearMonth'], y=monthly['Sales'], name="Total Sales ($)",
            line=dict(color="#1f77b4", width=3), mode='lines+markers',
            hovertemplate='<b>%{x}</b><br>Sales: $%{y:,.2f}<extra></extra>'
        ),
        secondary_y=False,
    )
    fig_monthly.add_trace(
        gg.Bar(
            x=monthly['YearMonth'], y=monthly['Profit'], name="Total Profit ($)",
            marker_color="#2ca02c", opacity=0.6,
            hovertemplate='<b>%{x}</b><br>Profit: $%{y:,.2f}<extra></extra>'
        ),
        secondary_y=True,
    )
    sales_axis_max = monthly['Sales'].max() * 1.10
    profit_axis_max = max(monthly['Profit'].max() * 1.20, 0)
    fig_monthly.update_layout(
        title_text="<b>Monthly Temporal Sales & Profit Trends</b>",
        hovermode="x unified",
        template="plotly_white",
        xaxis_title="Month",
        height=500,
        yaxis=dict(
            title_text="Sales ($)",
            range=[0, sales_axis_max],
            tickprefix="$",
            ticksuffix="",
            separatethousands=True,
            tickformat=',.0f',
            automargin=True,
            autorange=False
        ),
        yaxis2=dict(
            title_text="Profit ($)",
            range=[0, profit_axis_max],
            overlaying="y",
            tickprefix="$",
            separatethousands=True,
            tickformat=',.0f',
            automargin=True,
            autorange=False,
            showgrid=False
        )
    )

    # ---------------------------------------------------------
    # 2. Category & Sub-Category Treemap & Bar Chart
    # ---------------------------------------------------------
    cat_subcat = df.groupby(['Category', 'Sub-Category']).agg({
        'Sales': 'sum',
        'Profit': 'sum',
        'Quantity': 'sum'
    }).reset_index()
    cat_subcat['Profit_Margin_%'] = (cat_subcat['Profit'] / cat_subcat['Sales']) * 100

    _tm_ids, _tm_labels, _tm_parents, _tm_values, _tm_z = [], [], [], [], []

    _total_sales = float(df['Sales'].sum())
    _total_profit = float(df['Profit'].sum())
    _tm_ids.append("All Products")
    _tm_labels.append("All Products")
    _tm_parents.append("")
    _tm_values.append(_total_sales)
    _tm_z.append(_total_profit / _total_sales * 100 if _total_sales > 0 else 0)

    for _cat in df['Category'].unique():
        _cat_mask = df['Category'] == _cat
        _cat_sales = float(df.loc[_cat_mask, 'Sales'].sum())
        _cat_profit = float(df.loc[_cat_mask, 'Profit'].sum())
        _cat_id = "All Products/" + str(_cat)
        _tm_ids.append(_cat_id)
        _tm_labels.append(str(_cat))
        _tm_parents.append("All Products")
        _tm_values.append(_cat_sales)
        _tm_z.append(_cat_profit / _cat_sales * 100 if _cat_sales > 0 else 0)
        _sc_group = df.loc[_cat_mask].groupby('Sub-Category').agg({'Sales': 'sum', 'Profit': 'sum'})
        for _sc_name, _sc_row in _sc_group.iterrows():
            _sc_sales = float(_sc_row['Sales'])
            _sc_profit = float(_sc_row['Profit'])
            _sc_id = _cat_id + "/" + str(_sc_name)
            _tm_ids.append(_sc_id)
            _tm_labels.append(str(_sc_name))
            _tm_parents.append(_cat_id)
            _tm_values.append(_sc_sales)
            _tm_z.append(_sc_profit / _sc_sales * 100 if _sc_sales > 0 else 0)

    fig_treemap = gg.Figure(gg.Treemap(
        ids=_tm_ids,
        labels=_tm_labels,
        parents=_tm_parents,
        values=_tm_values,
        marker=dict(
            colors=_tm_z,
            colorscale='RdYlGn',
            cmin=-5,
            cmax=40,
            cmid=8,
            showscale=True,
            colorbar=dict(title="Profit_Margin_%", ticksuffix="%")
        ),
        branchvalues='total',
        hovertemplate='<b>%{label}</b><br>Sales: $%{value:,.2f}<br>Margin: %{color:.2f}%<extra></extra>',
        textinfo="label+text+value"
    ))
    fig_treemap.update_layout(
        title_text="<b>Product Category & Sub-Category Sales Hierarchy (Colored by Profit Margin %)</b>",
        height=500,
        template="plotly_white"
    )

    # ---------------------------------------------------------
    # 3. Customer Segment Performance & Sales-to-Profit Ratio
    # ---------------------------------------------------------
    segment_perf = df.groupby('Segment').agg({
        'Sales': 'sum',
        'Profit': 'sum',
        'Order ID': 'nunique'
    }).reset_index()
    segment_perf['Profit_Margin_%'] = (segment_perf['Profit'] / segment_perf['Sales']) * 100
    segment_perf['Sales_to_Profit_Ratio'] = segment_perf['Sales'] / segment_perf['Profit']
    # Pre-format labels (avoids texttemplate placeholder rendering bugs on CDN plotly.js versions)
    segment_perf['Sales_Label'] = segment_perf['Sales'].map(lambda v: f"${v:,.0f}")
    segment_perf['Profit_Label'] = segment_perf['Profit'].map(lambda v: f"${v:,.0f}")
    segment_perf['Ratio_Label'] = segment_perf['Sales_to_Profit_Ratio'].map(lambda v: f"{v:.2f}")

    fig_segment = make_subplots(
        rows=1, cols=2,
        subplot_titles=(
            "Total Sales & Profit by Customer Segment",
            "Sales-to-Profit Ratio ($ Sales per $1 Profit)"
        ),
        shared_yaxes=False,
        horizontal_spacing=0.15
    )

    fig_segment.add_trace(
        gg.Bar(
            name="Sales",
            x=segment_perf['Segment'],
            y=segment_perf['Sales'],
            marker_color="#3366cc",
            text=segment_perf['Sales_Label'].tolist(),
            textposition='outside',
            texttemplate='%{text}',
            hovertemplate='<b>%{x}</b><br>Sales: $%{y:,.2f}<extra></extra>'
        ),
        row=1, col=1
    )
    fig_segment.add_trace(
        gg.Bar(
            name="Profit",
            x=segment_perf['Segment'],
            y=segment_perf['Profit'],
            marker_color="#109618",
            text=segment_perf['Profit_Label'].tolist(),
            textposition='outside',
            texttemplate='%{text}',
            hovertemplate='<b>%{x}</b><br>Profit: $%{y:,.2f}<extra></extra>'
        ),
        row=1, col=1
    )
    fig_segment.add_trace(
        gg.Bar(
            name="Sales-to-Profit Ratio",
            x=segment_perf['Segment'],
            y=segment_perf['Sales_to_Profit_Ratio'],
            marker_color="#ff9900",
            text=segment_perf['Ratio_Label'].tolist(),
            textposition='outside',
            texttemplate='%{text}',
            hovertemplate='<b>%{x}</b><br>Ratio: %{y:.2f}<extra></extra>'
        ),
        row=1, col=2
    )
    fig_segment.update_layout(
        height=450,
        barmode='group',
        template="plotly_white",
        title_text="<b>Customer Segment Operational Analysis</b>"
    )
    # Left column: Sales & Profit $ axis
    # BUGFIX: autorange=True doesn't reserve headroom for `textposition='outside'`
    # labels, so the tallest bar's label (and its neighbor) get clipped/overlapped.
    # Compute an explicit range with headroom, same approach used for the other
    # axes in this script (sales_axis_max, profit_axis_max, ratio_max).
    amount_axis_max = max(segment_perf['Sales'].max(), segment_perf['Profit'].max()) * 1.30
    fig_segment.update_yaxes(
        title_text="Amount ($)",
        row=1, col=1,
        range=[0, amount_axis_max],
        autorange=False,
        rangemode="nonnegative",
        separatethousands=True,
        tickprefix="$",
        tickformat=',.0f'
    )
    # Right column: S:P ratio axis (separate scale, 0 to max*1.30, explicit tickformat)
    ratio_max = segment_perf['Sales_to_Profit_Ratio'].max() * 1.30
    fig_segment.update_yaxes(
        title_text="$ Sales per $1 Profit",
        row=1, col=2,
        range=[0, ratio_max],
        tick0=0,
        dtick=1,
        tickformat='.2f',
        separatethousands=False,
        showgrid=True,
        autorange=False
    )

    # ---------------------------------------------------------
    # 4. Profitability vs. Discount Interactive Scatter Plot
    # ---------------------------------------------------------
    # Downsample marker size: Sales range is $0.44 to $22,638 so default px size is too extreme
    df['_Size'] = np.clip(df['Sales'] / df['Sales'].max() * 60 + 4, 4, 40)

    fig_discount = px.scatter(
        df,
        x='Discount',
        y='Profit',
        size='_Size',
        size_max=40,
        color='Category',
        hover_data={
            'Sub-Category': True,
            'Product Name': True,
            'Customer Name': True,
            'City': True,
            'Sales': ':,.2f',
            '_Size': False
        },
        title="<b>Impact of Discount Level on Order Profitability</b>",
        labels={'Discount': 'Discount Rate (0.0 = 0%, 0.8 = 80%)', 'Profit': 'Profit / Loss ($)'},
        opacity=0.7,
        range_x=[-0.02, 0.82],
        range_y=[df['Profit'].min() * 1.05, df['Profit'].max() * 1.05]
    )
    fig_discount.add_hline(y=0, line_dash="dash", line_color="red", annotation_text="Break-even Line",
                          annotation_position="bottom right")
    fig_discount.update_layout(height=500, template="plotly_white")
    fig_discount.update_xaxes(dtick=0.1, tickformat='.1f')
    fig_discount.update_yaxes(tickprefix="$", tickformat=',.0f', separatethousands=True)

    # ---------------------------------------------------------
    # 5. Outlier Analysis Visualization
    # ---------------------------------------------------------
    # Build outlier summary explicitly to avoid mixed-multiindex agg bugs
    outlier_rows = df['Sales_Outlier']
    normal_count = int((~outlier_rows).sum())
    outlier_count = int(outlier_rows.sum())
    normal_sales = df.loc[~outlier_rows, 'Sales'].sum()
    outlier_sales = df.loc[outlier_rows, 'Sales'].sum()
    outlier_summary = pd.DataFrame({
        'Is_Outlier': [False, True],
        'Count': [normal_count, outlier_count],
        'Count_Label': [f"{normal_count:,} orders", f"{outlier_count:,} orders"],
        'Total_Sales': [normal_sales, outlier_sales]
    })
    outlier_summary['Label'] = outlier_summary['Is_Outlier'].map(
        {True: 'Outliers (IQR Extreme)', False: 'Normal Transactions'}
    )

    fig_outliers = px.bar(
        outlier_summary,
        x='Label',
        y='Total_Sales',
        color='Label',
        text='Count_Label',
        title="<b>Sales Breakdown: Outliers vs Normal Transactions</b>",
        labels={'Total_Sales': 'Total Sales ($)', 'Label': ''},
        range_y=[0, max(normal_sales, outlier_sales) * 1.25],
        hover_data={'Count': ':,.0f', 'Total_Sales': ':,.2f'}
    )
    fig_outliers.update_layout(height=400, template="plotly_white")
    fig_outliers.update_traces(
        textposition='outside',
        texttemplate='%{text}'
    )
    fig_outliers.update_yaxes(
        tickprefix="$",
        tickformat=',.0f',
        separatethousands=True
    )

    # ---------------------------------------------------------
    # Generate Combined HTML Dashboard File
    # ---------------------------------------------------------
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Task 2: Store Sales & Profit Dynamic Dashboard</title>
        <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
        <style>
            body {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                margin: 0;
                padding: 20px;
                background-color: #f4f7f6;
                color: #333;
            }}
            .header {{
                background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
                color: white;
                padding: 25px;
                border-radius: 12px;
                margin-bottom: 25px;
                box-shadow: 0 4px 15px rgba(0,0,0,0.1);
            }}
            .header h1 {{
                margin: 0 0 10px 0;
                font-size: 28px;
            }}
            .header p {{
                margin: 0;
                opacity: 0.9;
                font-size: 15px;
            }}
            .metrics-container {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
                gap: 15px;
                margin-bottom: 25px;
            }}
            .metric-card {{
                background: white;
                padding: 20px;
                border-radius: 10px;
                box-shadow: 0 2px 8px rgba(0,0,0,0.05);
                border-left: 5px solid #2a5298;
            }}
            .metric-title {{
                font-size: 13px;
                color: #666;
                text-transform: uppercase;
                letter-spacing: 0.5px;
            }}
            .metric-value {{
                font-size: 24px;
                font-weight: bold;
                color: #1e3c72;
                margin-top: 5px;
            }}
            .chart-card {{
                background: white;
                border-radius: 12px;
                padding: 20px;
                margin-bottom: 25px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.05);
            }}
            @media print {{
                @page {{
                    size: A4 landscape;
                    margin: 12mm;
                }}
                .chart-card, .metrics-container {{
                    page-break-inside: avoid;
                    break-inside: avoid;
                }}
                body {{
                    background: white;
                }}
            }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>Store Sales & Profit Dynamic Analysis Dashboard</h1>
            <p>Task 2: Interactive Plotly Visualizations & Operational Performance Insights | ShadowFox Analytics</p>
        </div>

        <div class="metrics-container">
            <div class="metric-card" style="border-left-color: #1f77b4;">
                <div class="metric-title">Total Revenue / Sales</div>
                <div class="metric-value">${df['Sales'].sum():,.2f}</div>
            </div>
            <div class="metric-card" style="border-left-color: #2ca02c;">
                <div class="metric-title">Total Net Profit</div>
                <div class="metric-value">${df['Profit'].sum():,.2f}</div>
            </div>
            <div class="metric-card" style="border-left-color: #ff9900;">
                <div class="metric-title">Overall Profit Margin</div>
                <div class="metric-value">{(df['Profit'].sum()/df['Sales'].sum()*100):.2f}%</div>
            </div>
            <div class="metric-card" style="border-left-color: #9467bd;">
                <div class="metric-title">Sales-to-Profit Ratio</div>
                <div class="metric-value">{(df['Sales'].sum()/df['Profit'].sum()):.2f} : 1</div>
            </div>
            <div class="metric-card" style="border-left-color: #d62728;">
                <div class="metric-title">Total Orders Analyzed</div>
                <div class="metric-value">{len(df):,}</div>
            </div>
        </div>

        <div class="chart-card">
            {fig_monthly.to_html(full_html=False, include_plotlyjs=False)}
        </div>

        <div class="chart-card">
            {fig_treemap.to_html(full_html=False, include_plotlyjs=False)}
        </div>

        <div class="chart-card">
            {fig_segment.to_html(full_html=False, include_plotlyjs=False)}
        </div>

        <div class="chart-card">
            {fig_discount.to_html(full_html=False, include_plotlyjs=False)}
        </div>

        <div class="chart-card">
            {fig_outliers.to_html(full_html=False, include_plotlyjs=False)}
        </div>

        <script>
            // BUGFIX: Plotly only resizes charts on the browser 'resize' event,
            // which Chrome's "Print to PDF" does NOT fire. Without this, charts
            // can be captured at the wrong (sometimes near-zero) size, producing
            // blank/flat lines or squashed, overlapping labels in the exported PDF.
            window.addEventListener('beforeprint', function () {{
                document.querySelectorAll('.js-plotly-plot').forEach(function (gd) {{
                    Plotly.Plots.resize(gd);
                }});
            }});
        </script>

    </body>
    </html>
    """

    output_filename = "interactive_sales_profit_dashboard.html"
    with open(output_filename, 'w', encoding='utf-8') as f:
        f.write(html_content)
        
    print(f"Interactive Plotly Dashboard successfully created: {output_filename}")

if __name__ == "__main__":
    create_plotly_dashboard()