import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="Amazon Sales Data Dashboard", layout="wide")
st.title("\U0001F4CA Amazon Sales Data Analysis - Spring 2025")
st.sidebar.header("\U0001F50D Navigation")
option = st.sidebar.radio("Go to section:", [
    "\U0001F4C2 View Dataset & Statistical Analysis",
    "\U0001F4C8 Visualizations",
    "\U0001F4DD Show Conclusion",
    "📈 Regression Model",
    "ℹ Project Description"
])

st.sidebar.header("\U0001F4C2 Upload CSV File")
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    cat_cols = df.select_dtypes(include='object').columns.tolist()

    if option == "\U0001F4C2 View Dataset & Statistical Analysis":
        st.subheader("\U0001F4CB Dataset Preview")
        st.dataframe(df)

        if numeric_cols:
            selected_col = st.selectbox("Select numeric column for analysis", numeric_cols)
            col_data = df[selected_col].dropna()

            mean = np.mean(col_data)
            median = np.median(col_data)
            mode = col_data.mode().iloc[0] if not col_data.mode().empty else np.nan
            std_dev = np.std(col_data)
            q1 = np.percentile(col_data, 25)
            q3 = np.percentile(col_data, 75)
            iqr = q3 - q1
            skewness = stats.skew(col_data)

            st.subheader("\U0001F4CC Descriptive Statistics Summary")

            stats_df = pd.DataFrame({
                "Statistic": [
                    "Mean", "Median", "Mode", "Standard Deviation", 
                    "Q1 (25th percentile)", "Q3 (75th percentile)", 
                    "IQR", "Skewness"
                ],
                "Value": [
                    f"{mean:.2f}", f"{median:.2f}", f"{mode:.2f}", f"{std_dev:.2f}", 
                    f"{q1:.2f}", f"{q3:.2f}", f"{iqr:.2f}", 
                    "Positive" if skewness > 0 else "Negative" if skewness < 0 else "Symmetrical"
                ]
            })
            st.table(stats_df)
        else:
            st.warning("\u26A0 This dataset does not contain numeric columns.")
    elif option == "\U0001F4C8 Visualizations":
        if numeric_cols:
            st.sidebar.header("\U0001F4CA Visualization Settings")
            chart_type = st.sidebar.selectbox("Select chart type", [
                "Histogram", "Bar Chart", "Multiple Bar Chart", "Pie Chart",
                "Box Plot", "Component Bar Chart", "Line Graph", "Outlier Detection (IQR)"
            ])
            st.subheader("\U0001F4C8 Data Visualization")
            if chart_type == "Histogram":
                selected_col = st.selectbox("Select numeric column", numeric_cols)
                bins = st.sidebar.slider("Number of bins", 5, 50, 10)
                fig = px.histogram(df, x=selected_col, nbins=bins, title=f"Histogram of {selected_col}", width=600, height=400)
                st.plotly_chart(fig, use_container_width=True)
                st.write(f"*Interpretation:* This histogram shows the distribution of {selected_col}.")
                st.write(f"If the data appears symmetrical, it suggests an even distribution. If most of the data points are concentrated on the left, it's positively skewed. If they're concentrated on the right, it's negatively skewed.")
                st.write(f"The tallest bars indicate the most frequent values, while the shortest bars represent the least frequent ones.")

                counts, edges = np.histogram(df[selected_col], bins=bins)
                max_count_index = np.argmax(counts)
                min_count_index = np.argmin(counts)

                st.write(f"The bin with the highest frequency is: {edges[max_count_index]:.2f} - {edges[max_count_index+1]:.2f}")
                st.write(f"The bin with the lowest frequency is: {edges[min_count_index]:.2f} - {edges[min_count_index+1]:.2f}")

            elif chart_type == "Bar Chart":
                if cat_cols:
                    cat_col = st.selectbox("Categorical column", cat_cols)
                    count_data = df[cat_col].value_counts()
                    fig = px.bar(x=count_data.index, y=count_data.values, labels={'x': cat_col, 'y': 'Count'}, title=f"Bar Chart of {cat_col}", width=600, height=400)
                    st.plotly_chart(fig, use_container_width=True)
                    st.write(f"*Interpretation:* This bar chart displays the frequency of each category in the '{cat_col}' column.")
                    st.write(f"The category with the highest count is: {count_data.idxmax()}")
                    st.write(f"The category with the lowest count is: {count_data.idxmin()}")
                else:
                    st.warning("No categorical column available for bar chart.")

            elif chart_type == "Multiple Bar Chart":
                if len(numeric_cols) >= 2:
                    col1 = st.selectbox("First numeric column", numeric_cols, key="mbc1")
                    col2 = st.selectbox("Second numeric column", numeric_cols, key="mbc2")
                    if cat_cols:
                        group_col = st.selectbox("Categorical column for grouping", cat_cols)
                        fig = px.bar(df, x=group_col, y=[col1, col2], barmode='group', title=f"Comparison of {col1} and {col2} by {group_col}", width=600, height=400)
                        st.plotly_chart(fig, use_container_width=True)
                        grouped_df = df.groupby(group_col)[[col1, col2]].sum()
                        st.write(f"The category with the highest combined value is {grouped_df.sum(axis=1).idxmax()}")
                        st.write(f"The category with the lowest combined value is {grouped_df.sum(axis=1).idxmin()}")
                    else:
                        st.warning("Please upload a dataset with at least one categorical column.")
                else:
                    st.warning("Please select at least two numeric columns.")

            elif chart_type == "Pie Chart":
                if cat_cols:
                    cat_col = st.selectbox("Pie column", cat_cols)
                    pie_data = df[cat_col].value_counts()
                    fig = px.pie(names=pie_data.index, values=pie_data.values, title=f"Distribution of {cat_col}", width=600, height=400)
                    st.plotly_chart(fig, use_container_width=True)
                    st.write(f"The category with the highest proportion is: {pie_data.idxmax()}")
                    st.write(f"The category with the lowest proportion is: {pie_data.idxmin()}")
                else:
                    st.warning("No categorical column available for pie chart.")

            elif chart_type == "Box Plot":
                selected_col = st.selectbox("Select numeric column", numeric_cols, key="boxplot")
                fig = px.box(df, y=selected_col, title=f"Box Plot of {selected_col}", width=600, height=400)
                st.plotly_chart(fig, use_container_width=True)

            elif chart_type == "Component Bar Chart":
                if len(cat_cols) >= 2 and len(numeric_cols) >= 1:
                    x_col = st.selectbox("Categorical column for x-axis", cat_cols, key="cbc_x")
                    color_col = st.selectbox("Component (color) category", cat_cols, key="cbc_color")
                    y_col = st.selectbox("Numeric column", numeric_cols, key="cbc_y")

                    if x_col == color_col:
                        st.error("Please select two different categorical columns.")
                    else:
                        fig = px.bar(df, x=x_col, y=y_col, color=color_col, title=f"Component Bar Chart: {y_col} by {x_col}", width=600, height=400)
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("Need at least 2 categorical and 1 numeric column.")

            elif chart_type == "Line Graph":
                selected_col = st.selectbox("Numeric column for line graph", numeric_cols, key="linegraph")
                fig = px.line(df, y=selected_col, title=f"Trend of {selected_col} over Records", width=600, height=400)
                st.plotly_chart(fig, use_container_width=True)

            elif chart_type == "Outlier Detection (IQR)":
                selected_col = st.selectbox("Select numeric column", numeric_cols, key="outlier")
                col_data = df[selected_col].dropna()
                q1 = np.percentile(col_data, 25)
                q3 = np.percentile(col_data, 75)
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                outliers = df[(df[selected_col] < lower_bound) | (df[selected_col] > upper_bound)]

                fig = px.box(df, y=selected_col, points="all", title=f"Outlier Detection using IQR for {selected_col}", width=600, height=400)
                st.plotly_chart(fig, use_container_width=True)
                st.write(f"Number of outliers detected: {outliers.shape[0]}")

                if not outliers.empty:
                    st.subheader("\U0001F4CC Outlier Details")
                    display_cols = [col for col in ['Product ID', 'Customer Name', 'Price', selected_col] if col in df.columns]
                    st.dataframe(outliers[display_cols] if display_cols else outliers)
                else:
                    st.write("\u2705 No outliers detected.")
        else:
            st.warning("\u26A0 This dataset does not contain numeric columns.")

    elif option == "\U0001F4DD Show Conclusion":
        st.subheader("\U0001F4DD Conclusion")
        st.write("""
         The Amazon Sales Data Dashboard provides a comprehensive overview of business operations and customer behavior through interactive analysis.

        ### 📌 Statistical Summary:
        - Central tendency measures (**mean**, **median**, **mode**) indicate typical values such as average prices or order quantities.
        - **Standard deviation**, **variance**, and **IQR** highlight variability, showing how consistent or scattered the data is.
        - **Skewness** reveals whether sales, prices, or quantities are skewed toward high or low extremes, useful in detecting anomalies or market trends.

        ### 📊 Visualization Insights:
        - **Histograms** offer insight into data distribution and shape.
        - **Bar and Pie Charts** highlight category-wise performance, customer behavior, and product popularity.
        - **Box Plots** expose the data's spread and detect outliers effectively.
        - **Line Graphs** help spot patterns, seasonality, or performance dips.
        - **Component and Multiple Bar Charts** are ideal for comparative analysis between different numeric fields across categories.
        - **Outlier Detection** using the IQR method identifies unusual data entries, such as abnormally high purchases or pricing errors.

        ### 📈 Regression Analysis:
        - The **Linear Regression** model helps understand relationships between variables like price and sales.
        - **R² scores** indicate how well one variable predicts another, guiding pricing strategies and sales forecasts.
        - Regression lines visualize trends and enable predictions based on historical data patterns.

        ### 💼 Business Applications:
        - Track **top-performing products** and underperformers.
        - Spot **customer behavior patterns**, high-value orders, or anomalies.
        - Drive **inventory decisions**, **pricing strategies**, and **marketing campaigns** based on trends and category strength.
        - Use findings to **optimize restocking**, **product bundling**, or **seasonal offers**.
        - Make **data-driven predictions** about future sales based on regression models.

        Overall, this dashboard bridges the gap between raw sales data and actionable business insights.
        """)
    
    elif option == "📈 Regression Model":
        st.subheader("📈 Simple Linear Regression")
        if len(numeric_cols) >= 2:
            x_col = st.selectbox("Select independent variable (X)", numeric_cols, key="reg_x")
            y_col = st.selectbox("Select dependent variable (Y)", [col for col in numeric_cols if col != x_col], key="reg_y")
            reg_df = df[[x_col, y_col]].dropna()
            X = reg_df[[x_col]]
            y = reg_df[y_col]
            model = LinearRegression()
            model.fit(X, y)
            y_pred = model.predict(X)
            r2 = model.score(X, y)
            
            # Calculate MSE and RMSE
            mse = np.mean((y - y_pred) ** 2)
            rmse = np.sqrt(mse)
            
            # Calculate correlation
            correlation = np.corrcoef(reg_df[x_col], reg_df[y_col])[0, 1]
            
            # Create two columns for metrics and interpretation
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Model Statistics")
                st.write(f"**Regression Equation:** {y_col} = {model.coef_[0]:.4f} × {x_col} + {model.intercept_:.4f}")
                st.write(f"**R² Score:** {r2:.4f}")
                st.write(f"**Correlation:** {correlation:.4f}")
                st.write(f"**Mean Squared Error (MSE):** {mse:.4f}")
                st.write(f"**Root Mean Squared Error (RMSE):** {rmse:.4f}")
            
            with col2:
                st.subheader("Interpretation")
                # Interpret coefficient
                if model.coef_[0] > 0:
                    coef_interpretation = f"As {x_col} increases, {y_col} tends to increase. For each unit increase in {x_col}, {y_col} increases by {model.coef_[0]:.4f} units on average."
                elif model.coef_[0] < 0:
                    coef_interpretation = f"As {x_col} increases, {y_col} tends to decrease. For each unit increase in {x_col}, {y_col} decreases by {abs(model.coef_[0]):.4f} units on average."
                else:
                    coef_interpretation = f"There appears to be no linear relationship between {x_col} and {y_col}."
                
                # Interpret R²
                if r2 < 0.3:
                    r2_interpretation = f"The R² value of {r2:.4f} indicates a weak relationship. Only {r2*100:.1f}% of the variation in {y_col} is explained by {x_col}."
                elif r2 < 0.7:
                    r2_interpretation = f"The R² value of {r2:.4f} indicates a moderate relationship. About {r2*100:.1f}% of the variation in {y_col} is explained by {x_col}."
                else:
                    r2_interpretation = f"The R² value of {r2:.4f} indicates a strong relationship. About {r2*100:.1f}% of the variation in {y_col} is explained by {x_col}."
                
                st.write(coef_interpretation)
                st.write(r2_interpretation)
                st.write(f"The RMSE value of {rmse:.4f} represents the typical prediction error in the same units as {y_col}.")
            
            # Create scatter plot with regression line
            fig = px.scatter(reg_df, x=x_col, y=y_col, 
                            title=f"Linear Regression: {y_col} vs {x_col}",
                            labels={x_col: x_col, y_col: y_col})
            
            # Add regression line
            fig.add_trace(go.Scatter(
                x=reg_df[x_col], 
                y=y_pred, 
                mode='lines', 
                name="Regression Line",
                line=dict(color='red', width=2)
            ))
            
            # Add confidence interval bands (optional visual enhancement)
            fig.update_layout(
                showlegend=True,
                height=500,
                hovermode='closest'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Add residuals plot
            residuals = y - y_pred
            fig_residuals = px.scatter(
                x=y_pred, 
                y=residuals,
                labels={"x": "Predicted Values", "y": "Residuals"},
                title="Residual Plot"
            )
            
            # Add horizontal line at y=0
            fig_residuals.add_hline(
                y=0, line_dash="dash", 
                line_color="red", 
                annotation_text="Zero Line",
                annotation_position="bottom right"
            )
            
            st.plotly_chart(fig_residuals, use_container_width=True)
            
            st.write("""
            ### How to interpret the residual plot:
            - **Pattern of dots:** Ideally, points should be randomly scattered around the zero line.
            - **Funnel shape:** If residuals fan out/in, it suggests heteroscedasticity (non-constant variance).
            - **Curve pattern:** A curved pattern suggests a non-linear relationship between variables.
            - **Random scatter:** Indicates a good linear fit to the data.
            """)
            
            # Option to download model predictions
            if st.checkbox("Show prediction data"):
                prediction_df = pd.DataFrame({
                    x_col: reg_df[x_col],
                    y_col: reg_df[y_col],
                    f"Predicted {y_col}": y_pred,
                    "Residuals": residuals
                })
                st.dataframe(prediction_df)
                
                # Create a CSV download option
                csv = prediction_df.to_csv(index=False)
                st.download_button(
                    label="Download predictions as CSV",
                    data=csv,
                    file_name=f"regression_predictions_{x_col}_{y_col}.csv",
                    mime="text/csv"
                )
        else:
            st.warning("Please upload a dataset with at least two numeric columns.")

    elif option == "ℹ Project Description":
        st.subheader("ℹ Project Description")
        st.write("""
       3. Data Description:
          Amazon Sales Data Analysis Project: Statistical Methods and Visualization

3.1  Dataset Description
The dataset contains information about Amazon orders from February to April 2025, including details such as:
- Order IDs and dates
- Product information (name, category, price)
- Transaction details (quantity, total sales)
- Customer information (name, location)
- Payment methods
- Order status (completed, pending, cancelled)

With 250 records and 11 variables, this dataset provides sufficient complexity for meaningful statistical analysis while remaining manageable for in-depth exploration.
3.2   Project Components
  3.2.1  Exploratory Data Analysis (EDA)
- *Bar Charts*: Visualize categorical variables such as product categories, customer locations, and payment methods to identify the most common values.
- *Histograms*: Analyze the distribution of continuous variables like price, quantity, and total sales to understand their central tendency and spread.
- *Pie Charts*: Represent proportional relationships between different categories, such as the market share of each product category or the distribution of order statuses.
- *Box Plots*: Identify the median, quartiles, and potential outliers in numerical variables, particularly focusing on price and sales distributions across different product categories.
 3.2.2. Statistical Analysis
- *Descriptive Statistics*: Calculate measures of central tendency (mean, median, mode) and dispersion (standard deviation, range, IQR) for key numerical variables.
- *Outlier Detection*: Use statistical methods such as Z-scores, IQR method, and visualization techniques to identify and analyze unusual observations in the dataset.
- *Probability Analysis*: Calculate various probabilities related to sales events, such as:
  - Probability of a sale being completed based on payment method
  - Probability of purchasing specific products given customer location
  - Conditional probabilities related to order status and product categories
3. Advanced Statistical Methods
- *Linear Regression Analysis*: Develop models to:
  - Predict total sales based on product features
  - Analyze the relationship between quantity ordered and price
  - Identify factors that significantly influence sales performance
- *Multiple Bar Charts*: Compare multiple variables simultaneously, such as sales across different product categories by location or payment method.
- *Bayes' Theorem Application*: Apply Bayesian inference to:
  - Calculate the probability of order completion given specific factors
  - Update probability estimates based on new evidence
  - Develop a probabilistic model for customer purchasing patterns

4. Interpretation and Business Insights
- *Sales Performance Analysis*: Identify top-performing products, categories, and locations.
- *Customer Behavior Analysis*: Discover patterns in customer preferences and purchasing habits.
- *Payment Method Analysis*: Determine the relationship between payment methods and order completion rates.
- *Geographical Analysis*: Analyze regional variations in sales patterns and product preferences.

5. Conclusion and Recommendations
- Summarize key findings from the statistical analyses
- Identify actionable insights for business strategy
- Propose recommendations for inventory management, marketing focus, and customer engagement
- Discuss limitations of the current analysis and potential areas for future investigation
6.  Technical Implementation
The project will be implemented using:
- Python programming language with libraries such as Pandas,SciPy,Stramlit and NumPy for data manipulation and statistical calculations
- Visualization libraries including Plotly for creating informative and visually appealing charts
- Stats models for regression analysis and other statistical modeling
7. Expected Outcomes
1. A comprehensive understanding of Amazon sales patterns and trends
2. Identification of factors that significantly influence sales performance
3. Probability models that can predict order completion and customer behavior
4. Visualizations that effectively communicate complex statistical insights
5. Actionable recommendations based on data-driven findings
8. Project Value
This analysis will provide valuable insights for e-commerce strategy, helping to optimize inventory management, tailor marketing efforts, and enhance customer experience based on statistical evidence rather than intuition. The project demonstrates the practical application of probability and statistics concepts in real-world business scenarios, showcasing how data analysis can drive informed decision-making and competitive advantage in the e-commerce sector.

        """)
else:
    st.info("\U0001F4C1 Please upload a CSV file to begin analysis.")