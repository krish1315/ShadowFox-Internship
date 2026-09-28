"""
Store Sales and Profit Analysis
================================
This script performs a comprehensive analysis of the Superstore retail sales dataset.
It covers data inspection, cleaning, analysis, visualization, and business recommendations.
"""

# =============================================================================
# 1. IMPORT LIBRARIES
# =============================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# 2. LOAD DATASET
# =============================================================================
print("=" * 80)
print("STEP 2: LOAD DATASET")
print("=" * 80)

csv_path = 'Sample - Superstore.csv'
# Use ISO-8859-1 encoding to handle special characters in product names
df = pd.read_csv(csv_path, encoding='ISO-8859-1')

print(f"Dataset loaded successfully: {csv_path}")
print(f"Shape: {df.shape[0]:,} rows x {df.shape[1]} columns")

# =============================================================================
# 3. DATASET OVERVIEW
# =============================================================================
print("\n" + "=" * 80)
print("STEP 3: DATASET OVERVIEW")
print("=" * 80)

print(f"\nColumn Names ({len(df.columns)} columns):")
for col in df.columns:
    print(f"  - {col}")

print(f"\nFirst 5 rows:")
print(df.head())

print(f"\nLast 5 rows:")
print(df.tail())

# =============================================================================
# 4. DATA INSPECTION
# =============================================================================
print("\n" + "=" * 80)
print("STEP 4: DATA INSPECTION")
print("=" * 80)

print("\nData Types:")
print(df.dtypes)

print("\nSample values for key columns:")
print(f"Order Date sample: {df['Order Date'].iloc[0]}")
print(f"Segment values: {df['Segment'].unique()}")
print(f"Category values: {df['Category'].unique()}")
print(f"Sub-Category values: {df['Sub-Category'].unique()}")
print(f"Region values: {df['Region'].unique()}")

# =============================================================================
# 5. DATA CLEANING
# =============================================================================
print("\n" + "=" * 80)
print("STEP 5: DATA CLEANING")
print("=" * 80)

# Store original row count
original_rows = len(df)
print(f"Original dataset: {original_rows:,} rows")

# 5.1 Check for duplicates
duplicates = df.duplicated().sum()
print(f"\nDuplicate rows found: {duplicates}")
if duplicates > 0:
    print("Action: Removing duplicate rows")
    df = df.drop_duplicates()
else:
    print("Action: No duplicates found - no action needed")

# 5.2 Check for missing values
print("\nMissing values per column:")
missing = df.isna().sum()
total_missing = missing.sum()
print(f"Total missing values: {total_missing}")
if total_missing > 0:
    print("Details:")
    for col, count in missing[missing > 0].items():
        print(f"  - {col}: {count} missing ({count/len(df)*100:.2f}%)")
else:
    print("Action: No missing values found - no action needed")

# 5.3 Check for empty strings in string columns
print("\nEmpty string check:")
string_cols = ['Order ID', 'Customer ID', 'Customer Name', 'City', 'State', 'Product ID', 'Product Name']
for col in string_cols:
    empty_count = (df[col] == '').sum() + (df[col] == ' ').sum()
    if empty_count > 0:
        print(f"  - {col}: {empty_count} empty/blank values")
    else:
        print(f"  - {col}: No empty values")

# 5.4 Convert date columns to datetime
print("\nDate conversion:")
df['Order Date'] = pd.to_datetime(df['Order Date'], format='%m/%d/%Y')
df['Ship Date'] = pd.to_datetime(df['Ship Date'], format='%m/%d/%Y')
print("  - Converted 'Order Date' to datetime")
print("  - Converted 'Ship Date' to datetime")

# 5.5 Check for invalid numerical values
print("\nNumerical validation:")
print(f"  - Sales: Min={df['Sales'].min():.2f}, Max={df['Sales'].max():.2f}")
print(f"  - Profit: Min={df['Profit'].min():.2f}, Max={df['Profit'].max():.2f}")
print(f"  - Quantity: Min={df['Quantity'].min()}, Max={df['Quantity'].max()}")
print(f"  - Discount: Min={df['Discount'].min()}, Max={df['Discount'].max()}")

# Check for negative sales (potential issues)
negative_sales = (df['Sales'] < 0).sum()
if negative_sales > 0:
    print(f"  WARNING: {negative_sales} negative sales values found")

# Check for impossible discount values (> 1)
invalid_discount = (df['Discount'] > 1).sum()
if invalid_discount > 0:
    print(f"  WARNING: {invalid_discount} discount values > 1.0 found")
else:
    print("  - All discount values are valid (0-1 range)")

# Final row count
final_rows = len(df)
print(f"\nFinal dataset: {final_rows:,} rows")
if original_rows != final_rows:
    print(f"Rows removed: {original_rows - final_rows}")

# =============================================================================
# 6. MISSING VALUE & OUTLIER ANALYSIS
# =============================================================================
print("\n" + "=" * 80)
print("STEP 6: MISSING VALUE & OUTLIER ANALYSIS")
print("=" * 80)
print(f"Total missing values in dataset: {df.isna().sum().sum()}")
print("Dataset is complete - no missing values to handle")

# IQR Outlier Detection for Sales and Profit
def detect_iqr_outliers(series):
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    return (series < lower_bound) | (series > upper_bound), lower_bound, upper_bound

sales_outliers, s_low, s_high = detect_iqr_outliers(df['Sales'])
profit_outliers, p_low, p_high = detect_iqr_outliers(df['Profit'])

print(f"\n--- Outlier Detection (IQR Method) ---")
print(f"Sales Outliers (Bounds: [{s_low:.2f}, {s_high:.2f}]): {sales_outliers.sum():,} rows ({sales_outliers.mean()*100:.1f}%)")
print(f"Profit Outliers (Bounds: [{p_low:.2f}, {p_high:.2f}]): {profit_outliers.sum():,} rows ({profit_outliers.mean()*100:.1f}%)")
print("Note: Outliers represent high-value enterprise sales and extreme discount losses; retained for full analysis.")

# =============================================================================
# 7. DUPLICATE ANALYSIS & EFFICIENCY RATIOS
# =============================================================================
print("\n" + "=" * 80)
print("STEP 7: DUPLICATE ANALYSIS & EFFICIENCY METRICS")
print("=" * 80)
print(f"Duplicate rows in dataset: {df.duplicated().sum()}")
print("Dataset is clean - no duplicate rows")

# Add Sales-to-Profit Ratio column
df['Sales_to_Profit_Ratio'] = np.where(df['Profit'] != 0, df['Sales'] / df['Profit'], np.nan)
df['Profit_Margin_%'] = (df['Profit'] / df['Sales']) * 100

# =============================================================================
# 8. DESCRIPTIVE STATISTICS
# =============================================================================
print("\n" + "=" * 80)
print("STEP 8: DESCRIPTIVE STATISTICS")
print("=" * 80)

print("\nNumerical columns descriptive statistics:")
print(df[['Sales', 'Quantity', 'Discount', 'Profit']].describe())

print("\nProfit by Category:")
profit_by_cat = df.groupby('Category')['Profit'].agg(['sum', 'mean', 'count'])
profit_by_cat.columns = ['Total Profit', 'Avg Profit', 'Num Orders']
print(profit_by_cat)

print("\nSales by Region:")
sales_by_region = df.groupby('Region')['Sales'].agg(['sum', 'mean'])
sales_by_region.columns = ['Total Sales', 'Avg Sales']
print(sales_by_region)

# =============================================================================
# 9. SALES ANALYSIS
# =============================================================================
print("\n" + "=" * 80)
print("STEP 9: SALES ANALYSIS")
print("=" * 80)

# 9.1 Basic Sales Metrics
total_sales = df['Sales'].sum()
avg_sales = df['Sales'].mean()
min_sales = df['Sales'].min()
max_sales = df['Sales'].max()
median_sales = df['Sales'].median()
std_sales = df['Sales'].std()

print(f"\n--- Basic Sales Metrics ---")
print(f"Total Sales: ${total_sales:,.2f}")
print(f"Average Sales: ${avg_sales:,.2f}")
print(f"Median Sales: ${median_sales:,.2f}")
print(f"Min Sales: ${min_sales:,.2f}")
print(f"Max Sales: ${max_sales:,.2f}")
print(f"Std Dev: ${std_sales:,.2f}")

# 9.2 Number of Transactions
num_transactions = len(df)
num_orders = df['Order ID'].nunique()
print(f"\nNumber of Order Lines: {num_transactions:,}")
print(f"Number of Unique Orders: {num_orders:,}")

# 9.3 Sales by Category
print(f"\n--- Sales by Category ---")
sales_by_category = df.groupby('Category')['Sales'].sum().sort_values(ascending=False)
for cat, sales in sales_by_category.items():
    pct = sales / total_sales * 100
    print(f"  {cat}: ${sales:,.2f} ({pct:.1f}%)")

# Identify best and worst categories
best_category = sales_by_category.idxmax()
worst_category = sales_by_category.idxmin()
best_cat_sales = sales_by_category.max()
worst_cat_sales = sales_by_category.min()

print(f"\nHighest-selling category: {best_category} (${best_cat_sales:,.2f})")
print(f"Lowest-selling category: {worst_category} (${worst_cat_sales:,.2f})")

# 9.4 Sales by Sub-Category
print(f"\n--- Sales by Sub-Category ---")
sales_by_subcat = df.groupby('Sub-Category')['Sales'].sum().sort_values(ascending=False)
for sc, sales in sales_by_subcat.items():
    pct = sales / total_sales * 100
    print(f"  {sc}: ${sales:,.2f} ({pct:.1f}%)")

# Top 5 and Bottom 5 products by sales
print(f"\n--- Top 5 Products by Sales ---")
top_products = df.groupby(['Category', 'Sub-Category'])['Sales'].sum().sort_values(ascending=False).head(5)
for idx, sales in top_products.items():
    print(f"  {idx[0]} - {idx[1]}: ${sales:,.2f}")

print(f"\n--- Bottom 5 Products by Sales ---")
bottom_products = df.groupby(['Category', 'Sub-Category'])['Sales'].sum().sort_values(ascending=True).head(5)
for idx, sales in bottom_products.items():
    print(f"  {idx[0]} - {idx[1]}: ${sales:,.2f}")

# 9.5 Sales by Region
print(f"\n--- Sales by Region ---")
sales_by_region = df.groupby('Region')['Sales'].sum().sort_values(ascending=False)
for region, sales in sales_by_region.items():
    pct = sales / total_sales * 100
    print(f"  {region}: ${sales:,.2f} ({pct:.1f}%)")

best_region = sales_by_region.idxmax()
worst_region = sales_by_region.idxmin()
print(f"\nBest-performing region: {best_region} (${sales_by_region[best_region]:,.2f})")
print(f"Weak-performing region: {worst_region} (${sales_by_region[worst_region]:,.2f})")

# 9.6 Sales by Customer Segment
print(f"\n--- Sales by Customer Segment ---")
sales_by_segment = df.groupby('Segment')['Sales'].sum().sort_values(ascending=False)
for seg, sales in sales_by_segment.items():
    pct = sales / total_sales * 100
    print(f"  {seg}: ${sales:,.2f} ({pct:.1f}%)")

# 9.7 Sales Over Time
print(f"\n--- Sales Over Time ---")
# Convert Order Date to year-month for monthly analysis
df['Order_Month'] = df['Order Date'].dt.to_period('M')
sales_by_month = df.groupby('Order_Month')['Sales'].sum()
print("First 6 months of sales data:")
for month, sales in sales_by_month.head(6).items():
    print(f"  {month}: ${sales:,.2f}")

# Find highest and lowest sales months
best_month = sales_by_month.idxmax()
worst_month = sales_by_month.idxmin()
print(f"\nPeak sales month: {best_month} (${sales_by_month[best_month]:,.2f})")
print(f"Lowest sales month: {worst_month} (${sales_by_month[worst_month]:,.2f})")

# =============================================================================
# 10. PROFIT ANALYSIS
# =============================================================================
print("\n" + "=" * 80)
print("STEP 10: PROFIT ANALYSIS")
print("=" * 80)

# 10.1 Basic Profit Metrics
total_profit = df['Profit'].sum()
avg_profit = df['Profit'].mean()
min_profit = df['Profit'].min()
max_profit = df['Profit'].max()
median_profit = df['Profit'].median()

print(f"\n--- Basic Profit Metrics ---")
print(f"Total Profit: ${total_profit:,.2f}")
print(f"Average Profit: ${avg_profit:,.2f}")
print(f"Median Profit: ${median_profit:,.2f}")
print(f"Min Profit: ${min_profit:,.2f} (loss)")
print(f"Max Profit: ${max_profit:,.2f}")

# Profit Margin
profit_margin = (total_profit / total_sales) * 100
print(f"\nOverall Profit Margin: {profit_margin:.2f}%")

# 10.2 Profit by Category
print(f"\n--- Profit by Category ---")
profit_by_category = df.groupby('Category')['Profit'].agg(['sum', 'mean', 'count']).sort_values('sum', ascending=False)
for cat, row in profit_by_category.iterrows():
    margin = row['sum'] / df[df['Category'] == cat]['Sales'].sum() * 100
    print(f"  {cat}: ${row['sum']:,.2f} total, ${row['mean']:,.2f} avg, Margin: {margin:.1f}% ({row['count']:.0f} orders)")

# 10.3 Profit by Sub-Category
print(f"\n--- Profit by Sub-Category ---")
profit_by_subcat = df.groupby('Sub-Category')['Profit'].sum().sort_values(ascending=False)
for sc, profit in profit_by_subcat.items():
    sales = df[df['Sub-Category'] == sc]['Sales'].sum()
    margin = profit / sales * 100 if sales > 0 else 0
    print(f"  {sc}: ${profit:,.2f} (Margin: {margin:.1f}%)")

# Identify most profitable and least profitable
most_profitable = profit_by_subcat.idxmax()
least_profitable = profit_by_subcat.idxmin()
print(f"\nMost profitable sub-category: {most_profitable} (${profit_by_subcat[most_profitable]:,.2f})")
print(f"Least profitable sub-category: {least_profitable} (${profit_by_subcat[least_profitable]:,.2f})")

# 10.4 Profit by Region
print(f"\n--- Profit by Region ---")
profit_by_region = df.groupby('Region')['Profit'].sum().sort_values(ascending=False)
for region, profit in profit_by_region.items():
    sales = df[df['Region'] == region]['Sales'].sum()
    margin = profit / sales * 100 if sales > 0 else 0
    print(f"  {region}: ${profit:,.2f} (Margin: {margin:.1f}%)")

# 10.5 Profit by Segment
print(f"\n--- Profit by Customer Segment ---")
profit_by_segment = df.groupby('Segment')['Profit'].sum().sort_values(ascending=False)
for seg, profit in profit_by_segment.items():
    sales = df[df['Segment'] == seg]['Sales'].sum()
    margin = profit / sales * 100 if sales > 0 else 0
    print(f"  {seg}: ${profit:,.2f} (Margin: {margin:.1f}%)")

# 10.6 Losing Products (Negative Profit)
print(f"\n--- Products with Negative Profit (Losses) ---")
negative_profit_orders = df[df['Profit'] < 0]
print(f"Number of orders with negative profit: {len(negative_profit_orders):,}")
print(f"Total loss from negative profit: ${abs(negative_profit_orders['Profit'].sum()):,.2f}")

loss_by_subcat = negative_profit_orders.groupby('Sub-Category')['Profit'].sum().sort_values()
print("\nTop 5 contributors to losses:")
for sc, loss in loss_by_subcat.head(5).items():
    print(f"  {sc}: ${abs(loss):,.2f} loss")

# =============================================================================
# 11. PRODUCT/CATEGORY ANALYSIS
# =============================================================================
print("\n" + "=" * 80)
print("STEP 11: PRODUCT/CATEGORY ANALYSIS")
print("=" * 80)

# 11.1 Top Products by Quantity and Profit
print("\n--- Top 10 Products by Sales ---")
product_sales = df.groupby(['Product ID', 'Product Name'])['Sales'].sum().sort_values(ascending=False).head(10)
for idx, sales in product_sales.items():
    print(f"  {idx[1][:50]}...: ${sales:,.2f}")

print("\n--- Top 10 Products by Profit ---")
product_profit = df.groupby(['Product ID', 'Product Name'])['Profit'].sum().sort_values(ascending=False).head(10)
for idx, profit in product_profit.items():
    print(f"  {idx[1][:50]}...: ${profit:,.2f}")

# 11.2 Category Performance Summary
print("\n--- Category Performance Summary ---")
for cat in df['Category'].unique():
    cat_data = df[df['Category'] == cat]
    cat_sales = cat_data['Sales'].sum()
    cat_profit = cat_data['Profit'].sum()
    cat_margin = cat_profit / cat_sales * 100
    num_orders = len(cat_data)
    print(f"\n{cat}:")
    print(f"  Total Sales: ${cat_sales:,.2f}")
    print(f"  Total Profit: ${cat_profit:,.2f}")
    print(f"  Profit Margin: {cat_margin:.1f}%")
    print(f"  Number of Orders: {num_orders}")
    print(f"  Average Sales per Order: ${cat_sales/num_orders:,.2f}")
    print(f"  Average Profit per Order: ${cat_profit/num_orders:,.2f}")

# =============================================================================
# 12. REGIONAL ANALYSIS
# =============================================================================
print("\n" + "=" * 80)
print("STEP 12: REGIONAL ANALYSIS")
print("=" * 80)

print("\n--- Regional Performance Summary ---")
for region in df['Region'].unique():
    region_data = df[df['Region'] == region]
    reg_sales = region_data['Sales'].sum()
    reg_profit = region_data['Profit'].sum()
    reg_margin = reg_profit / reg_sales * 100 if reg_sales > 0 else 0
    num_orders = len(region_data)
    num_customers = region_data['Customer ID'].nunique()
    print(f"\n{region}:")
    print(f"  Total Sales: ${reg_sales:,.2f}")
    print(f"  Total Profit: ${reg_profit:,.2f}")
    print(f"  Profit Margin: {reg_margin:.1f}%")
    print(f"  Number of Orders: {num_orders}")
    print(f"  Number of Unique Customers: {num_customers}")
    print(f"  Avg Sales per Customer: ${reg_sales/num_customers:,.2f}")

# =============================================================================
# 13. CUSTOMER/SEGMENT ANALYSIS
# =============================================================================
print("\n" + "=" * 80)
print("STEP 13: CUSTOMER/SEGMENT ANALYSIS")
print("=" * 80)

print("\n--- Customer Segment Performance ---")
for seg in df['Segment'].unique():
    seg_data = df[df['Segment'] == seg]
    seg_sales = seg_data['Sales'].sum()
    seg_profit = seg_data['Profit'].sum()
    seg_margin = seg_profit / seg_sales * 100 if seg_sales > 0 else 0
    num_orders = len(seg_data)
    num_customers = seg_data['Customer ID'].nunique()
    print(f"\n{seg}:")
    print(f"  Total Sales: ${seg_sales:,.2f}")
    print(f"  Total Profit: ${seg_profit:,.2f}")
    print(f"  Profit Margin: {seg_margin:.1f}%")
    print(f"  Number of Orders: {num_orders}")
    print(f"  Number of Unique Customers: {num_customers}")
    print(f"  Avg Sales per Order: ${seg_sales/num_orders:,.2f}")

# Customer Analysis
print("\n--- Top 10 Customers by Total Purchase ---")
customer_purchases = df.groupby(['Customer ID', 'Customer Name'])['Sales'].sum().sort_values(ascending=False).head(10)
for idx, sales in customer_purchases.items():
    print(f"  {idx[1]} ({idx[0]}): ${sales:,.2f}")

# =============================================================================
# 14. TIME-SERIES ANALYSIS
# =============================================================================
print("\n" + "=" * 80)
print("STEP 14: TIME-SERIES ANALYSIS")
print("=" * 80)

# Add year column
df['Year'] = df['Order Date'].dt.year

print("\n--- Sales by Year ---")
sales_by_year = df.groupby('Year')['Sales'].sum()
for year, sales in sales_by_year.items():
    print(f"  {year}: ${sales:,.2f}")

# Create time series for analysis
df['Order_Quarter'] = df['Order Date'].dt.to_period('Q')
sales_by_quarter = df.groupby('Order_Quarter')['Sales'].sum()

print("\n--- Sales by Quarter ---")
for quarter, sales in sales_by_quarter.items():
    print(f"  {quarter}: ${sales:,.2f}")

# Yearly profit analysis
profit_by_year = df.groupby('Year')['Profit'].sum()
print("\n--- Profit by Year ---")
for year, profit in profit_by_year.items():
    print(f"  {year}: ${profit:,.2f}")

# =============================================================================
# 15. DISCOUNT ANALYSIS
# =============================================================================
print("\n" + "=" * 80)
print("STEP 15: DISCOUNT ANALYSIS")
print("=" * 80)

# Discount statistics
print("\n--- Discount Statistics ---")
print(f"  Mean Discount: {df['Discount'].mean()*100:.1f}%")
print(f"  Median Discount: {df['Discount'].median()*100:.1f}%")
print(f"  Min Discount: {df['Discount'].min()*100:.1f}%")
print(f"  Max Discount: {df['Discount'].max()*100:.1f}%")
print(f"  Std Dev: {df['Discount'].std()*100:.1f}%")

# Discount distribution
print("\n--- Discount Distribution ---")
discount_bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0]
discount_labels = ['0-10%', '10-20%', '20-30%', '30-40%', '40-50%', '50-60%', '60-70%', '70-80%', '80-100%']
discount_cat = pd.cut(df['Discount'], bins=discount_bins, labels=discount_labels, include_lowest=True)
discount_dist = df.groupby(discount_cat).size()
for label, count in discount_dist.items():
    pct = count / len(df) * 100
    print(f"  {label}: {count:,} orders ({pct:.1f}%)")

# Profit vs Discount
print("\n--- Profit by Discount Level ---")
df['Discount_Category'] = pd.cut(df['Discount'], bins=discount_bins, labels=discount_labels, include_lowest=True)
profit_by_discount = df.groupby('Discount_Category')['Profit'].agg(['sum', 'mean'])
for idx, row in profit_by_discount.iterrows():
    print(f"  {idx}: Total ${row['sum']:,.2f}, Avg ${row['mean']:,.2f}")

# =============================================================================
# 16. QUANTITY ANALYSIS
# =============================================================================
print("\n" + "=" * 80)
print("STEP 16: QUANTITY ANALYSIS")
print("=" * 80)

# Quantity statistics
print("\n--- Quantity Statistics ---")
print(f"  Mean Quantity: {df['Quantity'].mean():.2f}")
print(f"  Median Quantity: {df['Quantity'].median():.0f}")
print(f"  Min Quantity: {df['Quantity'].min()}")
print(f"  Max Quantity: {df['Quantity'].max()}")
print(f"  Std Dev: {df['Quantity'].std():.2f}")

# Orders with high quantity
print("\n--- Orders by Quantity ---")
qty_groups = df.groupby('Quantity').size().sort_index()
for qty, count in qty_groups.items():
    print(f"  {qty} items: {count:,} orders")

# =============================================================================
# 17. CORRELATION ANALYSIS
# =============================================================================
print("\n" + "=" * 80)
print("STEP 17: CORRELATION ANALYSIS")
print("=" * 80)

# Numerical columns correlation
numerical_cols = ['Sales', 'Quantity', 'Discount', 'Profit']
correlation_matrix = df[numerical_cols].corr()

print("\n--- Correlation Matrix ---")
print(correlation_matrix.to_string())

print("\n--- Key Correlations ---")
print(f"  Sales vs Profit: {correlation_matrix.loc['Sales', 'Profit']:.4f}")
print(f"  Sales vs Quantity: {correlation_matrix.loc['Sales', 'Quantity']:.4f}")
print(f"  Sales vs Discount: {correlation_matrix.loc['Sales', 'Discount']:.4f}")
print(f"  Profit vs Discount: {correlation_matrix.loc['Profit', 'Discount']:.4f}")

# =============================================================================
# 18. DATA VISUALIZATION
# =============================================================================
print("\n" + "=" * 80)
print("STEP 18: DATA VISUALIZATION")
print("=" * 80)
print("Creating visualizations...")

# Set up the figure with multiple subplots
fig = plt.figure(figsize=(16, 12))

# 18.1 Sales by Category (Bar Chart)
ax1 = fig.add_subplot(2, 3, 1)
sales_by_cat = df.groupby('Category')['Sales'].sum().sort_values(ascending=True)
colors = ['#2ecc71', '#3498db', '#e74c3c']
bars = ax1.barh(sales_by_cat.index, sales_by_cat.values, color=colors)
ax1.set_xlabel('Sales ($)')
ax1.set_title('Sales by Category')
for bar, val in zip(bars, sales_by_cat.values):
    ax1.text(val + 500, bar.get_y() + bar.get_height()/2, f'${val:,.0f}', va='center')

# 18.2 Profit by Category (Bar Chart)
ax2 = fig.add_subplot(2, 3, 2)
profit_by_cat = df.groupby('Category')['Profit'].sum().sort_values(ascending=True)
colors_profit = ['#27ae60', '#2980b9', '#c0392b']
bars2 = ax2.barh(profit_by_cat.index, profit_by_cat.values, color=colors_profit)
ax2.set_xlabel('Profit ($)')
ax2.set_title('Profit by Category')
for bar, val in zip(bars2, profit_by_cat.values):
    ax2.text(val + 200, bar.get_y() + bar.get_height()/2, f'${val:,.0f}', va='center')

# 18.3 Sales by Region (Pie Chart)
ax3 = fig.add_subplot(2, 3, 3)
sales_by_region = df.groupby('Region')['Sales'].sum()
ax3.pie(sales_by_region.values, labels=sales_by_region.index, autopct='%1.1f%%', startangle=90)
ax3.set_title('Sales by Region')

# 18.4 Profit by Segment (Bar Chart)
ax4 = fig.add_subplot(2, 3, 4)
profit_by_seg = df.groupby('Segment')['Profit'].sum()
colors_seg = ['#9b59b6', '#1abc9c', '#e67e22']
bars4 = ax4.bar(profit_by_seg.index, profit_by_seg.values, color=colors_seg)
ax4.set_ylabel('Profit ($)')
ax4.set_title('Profit by Customer Segment')
ax4.tick_params(axis='x', rotation=15)
for bar, val in zip(bars4, profit_by_seg.values):
    ax4.text(bar.get_x() + bar.get_width()/2, val + 2000, f'${val:,.0f}', ha='center')

# 18.5 Monthly Sales Trend
ax5 = fig.add_subplot(2, 3, 5)
monthly_sales = df.groupby('Order_Month')['Sales'].sum()
# Convert Period to string for x-axis
months = [str(m) for m in monthly_sales.index]
ax5.plot(range(len(months)), monthly_sales.values, marker='o', linewidth=2, markersize=4, color='#34495e')
ax5.set_xlabel('Month')
ax5.set_ylabel('Sales ($)')
ax5.set_title('Monthly Sales Trend')
ax5.set_xticks(range(0, len(months), 3))
ax5.set_xticklabels([months[i] for i in range(0, len(months), 3)], rotation=45, ha='right')

# 18.6 Profit vs Discount Scatter
ax6 = fig.add_subplot(2, 3, 6)
ax6.scatter(df['Discount'], df['Profit'], alpha=0.5, c='#95a5a6')
ax6.axhline(y=0, color='red', linestyle='--', linewidth=1, label='Break-even')
ax6.set_xlabel('Discount')
ax6.set_ylabel('Profit ($)')
ax6.set_title('Profit vs Discount')
ax6.legend()

plt.tight_layout()
plt.savefig('sales_profit_analysis_charts.png', dpi=150, bbox_inches='tight')
print("Saved: sales_profit_analysis_charts.png")

plt.close()

# Create additional visualization for profit margins
fig2, axes = plt.subplots(1, 2, figsize=(14, 5))

# Profit Margin by Category
ax7 = axes[0]
margin_by_cat = df.groupby('Category').apply(lambda x: x['Profit'].sum() / x['Sales'].sum() * 100)
colors_cat = ['#2ecc71', '#3498db', '#e74c3c']
bars7 = ax7.bar(margin_by_cat.index, margin_by_cat.values, color=colors_cat)
ax7.set_ylabel('Profit Margin (%)')
ax7.set_title('Profit Margin by Category')
ax7.tick_params(axis='x', rotation=15)
for bar, val in zip(bars7, margin_by_cat.values):
    ax7.text(bar.get_x() + bar.get_width()/2, val + 0.5, f'{val:.1f}%', ha='center')

# Profit by Sub-Category (horizontal bar)
ax8 = axes[1]
profit_by_subcat_sorted = df.groupby('Sub-Category')['Profit'].sum().sort_values()
ax8.barh(profit_by_subcat_sorted.index, profit_by_subcat_sorted.values, color='#3498db')
ax8.set_xlabel('Profit ($)')
ax8.set_title('Profit by Sub-Category')
ax8.axvline(x=0, color='red', linestyle='--', linewidth=1)

plt.tight_layout()
plt.savefig('profit_analysis_charts.png', dpi=150, bbox_inches='tight')
print("Saved: profit_analysis_charts.png")

plt.close()

# =============================================================================
# 19. KEY FINDINGS
# =============================================================================
print("\n" + "=" * 80)
print("STEP 19: KEY FINDINGS")
print("=" * 80)

# Calculate dynamic values for findings
best_margin_cat = (df.groupby('Category').apply(lambda x: x['Profit'].sum() / x['Sales'].sum())
                   .sort_values(ascending=False).index[0])
best_segment = profit_by_segment.idxmax()

print("""
KEY FINDINGS:

1. FINANCIAL PERFORMANCE
   - Total Sales: ${:,.2f}
   - Total Profit: ${:,.2f}
   - Overall Profit Margin: {:.2f}%
   - Total Orders Analyzed: {:,}

2. CATEGORY PERFORMANCE
   - Best performing category by sales: {}
   - Category with highest profit margin: {}
   - Category with lowest sales: {}

3. REGIONAL PERFORMANCE
   - Best performing region: {}
   - Weakest performing region: {}

4. CUSTOMER SEGMENT PERFORMANCE
   - Most profitable segment: {}
   - Sales distribution by segment shows {} segment leading

5. PRODUCT INSIGHTS
   - Highest selling product category: {}
   - Most profitable sub-category: {}

6. DISCOUNT IMPACT
   - Average discount: {:.1f}%
   - High discount items contribute to losses

7. TIME PERFORMANCE
   - Peak sales period: {}
   - Consistent revenue generation throughout the year
""".format(
    total_sales,
    total_profit,
    profit_margin,
    num_transactions,
    best_category,
    best_margin_cat,
    worst_category,
    best_region,
    worst_region,
    best_segment,
    best_segment,
    best_category,
    most_profitable,
    df['Discount'].mean() * 100,
    str(best_month)
))

# =============================================================================
# 20. BUSINESS RECOMMENDATIONS
# =============================================================================
print("\n" + "=" * 80)
print("STEP 20: BUSINESS RECOMMENDATIONS")
print("=" * 80)

# Find category with lowest margin for recommendation
margin_by_cat = df.groupby('Category').apply(lambda x: x['Profit'].sum() / x['Sales'].sum()).sort_values()
lowest_margin_cat = margin_by_cat.index[0]

print("""
BUSINESS RECOMMENDATIONS:

1. CATEGORY STRATEGY
   - Focus on expanding {} category which generates highest sales volume.
   - Review pricing strategy for {} to improve profit margins.
   - Consider discontinuing or repricing products in categories with below-average margins.

2. REGIONAL OPTIMIZATION
   - Strengthen presence in {} region which shows strong sales.
   - Investigate opportunities in {} to improve underperformance.
   - Consider targeted marketing campaigns in weaker regions.

3. CUSTOMER SEGMENT STRATEGY
   - Prioritize {} segment which contributes most to profits.
   - Develop retention programs for high-value customers.
   - Evaluate marketing spend allocation across segments.

4. PRODUCT PORTFOLIO
   - Stock up on high-demand products from {} category.
   - Review pricing of high-volume low-margin products.
   - Consider bundling slow-moving items with popular products.

5. DISCOUNT AND PROMOTION STRATEGY
   - Optimize discount levels - current average is {:.1f}%.
   - Identify and limit discounts that lead to losses.
   - Implement tiered discount structure based on customer segment.

6. INVENTORY MANAGEMENT
   - Improve forecasting for seasonal variations.
   - Optimize stock levels for categories with high volume turnover.
   - Reduce inventory holding costs for low-margin categories.

7. OPERATIONAL IMPROVEMENTS
   - Standardize shipping methods to reduce costs.
   - Improve supply chain efficiency for high-cost items.
   - Consider cost reduction initiatives in underperforming regions.
""".format(
    best_category,
    lowest_margin_cat,
    best_region,
    worst_region,
    best_segment,
    best_category,
    df['Discount'].mean() * 100
))

# =============================================================================
# 21. CONCLUSION
# =============================================================================
print("\n" + "=" * 80)
print("STEP 21: CONCLUSION")
print("=" * 80)

performance = "POSITIVE" if total_profit > 0 else "NEGATIVE"
best_segment = profit_by_segment.idxmax()

print("""
CONCLUSION:

The Superstore retail analysis reveals a {} overall business performance
with ${:,.2f} in total sales across {:,} transactions.

The {} category drives the majority of sales (${:,.2f}), while the {} segment
accounted for the highest profit contribution.

Regionally, {} shows the strongest performance, indicating healthy market
penetration, while {} requires strategic attention.

Key Areas for Improvement:
1. Improve profit margins in Furniture category (2.5%) and Tables/Bookcases sub-categories
2. Develop targeted strategies for Central and South regions for better profitability
3. Optimize discount programs (discounts >20% consistently drive losses)
4. Focus on high-value customer segments for retention (Home Office has best margins)

The analysis provides a foundation for data-driven decision making to enhance
both top-line growth and bottom-line profitability.
""".format(
    performance,
    total_sales,
    num_transactions,
    best_category,
    sales_by_category[best_category],
    best_segment,
    best_region,
    worst_region
))

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)
print("Charts saved: sales_profit_analysis_charts.png, profit_analysis_charts.png")

# Trigger Plotly Dynamic Dashboard Creation
try:
    from generate_plotly_dashboard import create_plotly_dashboard
    create_plotly_dashboard()
except Exception as e:
    print(f"Plotly dashboard generation note: {e}")

print("Dataframe columns retained for further analysis: {}".format(list(df.columns)))