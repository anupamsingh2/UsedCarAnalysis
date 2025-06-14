import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy as sp

# Download from URL directly instead of local file
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/autos/imports-85.data"

#Step 5: Assign Column Headers
headers = [
    "symboling", "normalized-losses", "make", "fuel-type", "aspiration",
    "num-of-doors", "body-style", "drive-wheels", "engine-location",
    "wheel-base", "length", "width", "height", "curb-weight", "engine-type",
    "num-of-cylinders", "engine-size", "fuel-system", "bore", "stroke",
    "compression-ratio", "horsepower", "peak-rpm", "city-mpg", "highway-mpg", "price"
]

# Use headers directly in read_csv
df = pd.read_csv(url, names=headers)

print(df.head())

# Step 6: Check for Missing Values

data = df

# Replace all '?' strings with actual NaN values
data.replace('?', np.nan, inplace=True)

# Now check for missing values
print(data.isnull().any())

print(data.isnull().sum())


#Step 7: Convert MPG to L/100km
data['city-mpg'] = 235 / df['city-mpg']
data.rename(columns = {'city_mpg': "city-L / 100km"}, inplace = True)

print(data.columns)

print(data.dtypes)

#Step 8: Convert Price Column to Integer
data.price.unique()

# Drop rows where price is missing (was '?' before)
data = data.dropna(subset=['price'])

# Convert price column to integer
data['price'] = data['price'].astype(int)

print(data.dtypes)


#Step 9: Normalize Features

data['length'] = data['length']/data['length'].max()
data['width'] = data['width']/data['width'].max()
data['height'] = data['height']/data['height'].max()

# binning- grouping values
bins = np.linspace(min(data['price']), max(data['price']), 4) 
group_names = ['Low', 'Medium', 'High']
data['price-binned'] = pd.cut(data['price'], bins, labels = group_names, include_lowest = True)

print(data['price-binned'])
plt.hist(data['price-binned'])
plt.show()

#Step 10: Convert Categorical Data to Numerical
print(pd.get_dummies(data['fuel-type']).head())

print(data.describe())

#Step 11: Data Visualization
plt.boxplot(data['price'])

sns.boxplot(x ='drive-wheels', y ='price', data = data)

plt.scatter(data['engine-size'], data['price'])
plt.title('Scatterplot of Enginesize vs Price')
plt.xlabel('Engine size')
plt.ylabel('Price')
plt.grid()
plt.show()

#Step 12: Grouping Data by Drive-Wheels and Body-Style
test = data[['drive-wheels', 'body-style', 'price']]
data_grp = test.groupby(['drive-wheels', 'body-style'],as_index = False).mean()

print(data_grp)

#Step 13: Create a Pivot Table & Heatmap
data_pivot = data_grp.pivot(index = 'drive-wheels',columns = 'body-style')
print(data_pivot)

plt.pcolor(data_pivot, cmap ='RdBu')
plt.colorbar()
plt.show()

#Step 14: Perform ANOVA Test
data_annova = data[['make', 'price']]
grouped_annova = data_annova.groupby(['make'])
annova_results_l = sp.stats.f_oneway(grouped_annova.get_group('honda')['price'],grouped_annova.get_group('subaru')['price'])
print(annova_results_l)

sns.regplot(x ='engine-size', y ='price', data = data)
plt.ylim(0, )
plt.show()