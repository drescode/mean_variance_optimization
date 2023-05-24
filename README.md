# mean_variance_optimization

Stock Portfolio Optimization

This project focuses on the application of portfolio optimization techniques on selected stocks. The purpose is to understand the trade-off between risk and return and to identify optimal portfolio compositions using historical stock data.

Overview

The project involves several stages:

Loading and processing the stock data
Computing quarterly returns for each period
Calculating expected returns
Calculating the covariance matrix for the returns
Simulating a set of random portfolios
Determining and plotting the Efficient Frontier
Comparing the set of portfolios on the Efficient Frontier
Data

The stock data is obtained from a CSV file named 'stock_data3.csv'. It contains historical prices for multiple stocks. In this analysis, we specifically focus on five stocks: 'KO', 'HPQ', 'M', 'VZ', and 'MRO'.

Methods

We calculate the expected returns and covariance of returns for the selected stocks. This data is used to simulate a set of random portfolios, each with a different allocation of stocks. The random portfolios are then plotted to visualize the risk-return trade-off.

The Efficient Frontier is calculated using the 'optimal_portfolio' function, which determines the set of optimal portfolios offering the highest expected return for a defined level of risk. These optimal portfolios are then compared to the individual assets and the set of random portfolios.

Finally, we compare the Efficient Frontier of our selected five stocks with the Efficient Frontier of all ten stocks. This provides insights into how asset selection can impact portfolio optimization.

Requirements

This project is implemented in Python and requires the following libraries:

pandas
numpy
matplotlib
Results

The results of this project demonstrate the benefits of diversification and portfolio optimization. It shows that by carefully selecting the composition of a portfolio, one can maximize expected returns for a given level of risk.

Usage

To run the project, ensure that you have the required libraries installed in your Python environment. The main analysis can be run by executing the Python script or notebook containing the code.

Please make sure that 'stock_data3.csv' and 'all_ten.csv' are in the correct directory for the script to access.

Conclusion

This project serves as a practical application of portfolio theory, demonstrating how investors can use historical data to make informed decisions about portfolio allocation. The analysis highlights the benefits of diversification, the concept of the Efficient Frontier, and the impact of asset selection on portfolio optimization.
