import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class DataAnalyzer:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = None

    def load_data(self):
        self.data = pd.read_csv(self.file_path)
        return self.data.head()

    def rename_columns(self):
        self.data.columns = [
            'Company_Code', 'Company_Name', 'Revenue', 'Revenue_Growth_Rate', 'Net_Income_Loss', 
            'Net_Worth', 'Return_on_Equity', 'Stock', 'Total_Assets', 'Return_on_Assets', 
            'Asset_Turnover_Ratio', 'Debt_Ratio', 'EPS', 'Top_100'
        ]
        return self.data.head()

    def convert_columns_to_str(self):
        self.data = self.data.astype(str)

    def clean_numeric_column(self, column):
        return column.str.replace('[(),]', '', regex=True).astype(float)

    def convert_numerical_columns(self):
        self.data['Revenue'] = self.clean_numeric_column(self.data['Revenue'])
        self.data['Net_Income_Loss'] = self.clean_numeric_column(self.data['Net_Income_Loss'])
        self.data['Net_Worth'] = self.clean_numeric_column(self.data['Net_Worth'])
        self.data['Stock'] = self.clean_numeric_column(self.data['Stock'])
        self.data['Total_Assets'] = self.clean_numeric_column(self.data['Total_Assets'])
        self.data['Return_on_Equity'] = self.clean_numeric_column(self.data['Return_on_Equity'])
        self.data['Return_on_Assets'] = self.clean_numeric_column(self.data['Return_on_Assets'])
        self.data['Asset_Turnover_Ratio'] = self.clean_numeric_column(self.data['Asset_Turnover_Ratio'])
        self.data['Debt_Ratio'] = self.clean_numeric_column(self.data['Debt_Ratio'])
        self.data['EPS'] = self.clean_numeric_column(self.data['EPS'])
        self.data['Revenue_Growth_Rate'] = self.clean_numeric_column(self.data['Revenue_Growth_Rate'])
        return self.data.head()

    def handle_missing_values(self):
        self.data['Revenue_Growth_Rate'].fillna(self.data['Revenue_Growth_Rate'].mean(), inplace=True)
        self.data['Return_on_Equity'].fillna(self.data['Return_on_Equity'].mean(), inplace=True)
        self.data['Return_on_Assets'].fillna(self.data['Return_on_Assets'].mean(), inplace=True)
        self.data['Asset_Turnover_Ratio'].fillna(self.data['Asset_Turnover_Ratio'].mean(), inplace=True)
        self.data['Debt_Ratio'].fillna(self.data['Debt_Ratio'].mean(), inplace=True)
        self.data['Stock'].fillna(self.data['Stock'].mean(), inplace=True)
        self.data['Total_Assets'].fillna(self.data['Total_Assets'].mean(), inplace=True)
        return self.data.isnull().mean() * 100

    def check_duplicates(self):
        return self.data.duplicated().sum()

    def plot_correlation_matrix(self):
        correlation_matrix = self.data.corr()
        plt.figure(figsize=(12, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
        plt.title('Correlation Matrix')
        plt.show()

    def plot_histograms(self):
        self.data.hist(figsize=(16, 12), bins=30, edgecolor='black')
        plt.suptitle('Histogram of Numerical Columns')
        plt.show()

    def Drop_Company_Name_Column(self):
        self.data.drop(columns=['Company_Name'], inplace=True)
        return self.data.head())

# Usage
analyzer = DataAnalyzer('/mnt/data/Top100.csv')
analyzer.load_data()
analyzer.rename_columns()
analyzer.convert_columns_to_str()
analyzer.convert_numerical_columns()
analyzer.handle_missing_values()
duplicates = analyzer.check_duplicates()
analyzer.plot_correlation_matrix()
analyzer.plot_histograms()
analyzer.Drop_Company_Name_Column()

