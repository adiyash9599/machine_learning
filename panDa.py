import pandas as pd

# df = pd.read_csv('pokemon_data.csv')
# df = pd.read_excel('pokemon_data.xlsx')
df = pd.read_csv('pokemon_data.txt', delimiter='\t') # -> tab separated
# print(df)
# print(df.head(3))
# print(df.tail(3))

# read headers
# print(df.columns)

# print(df.columns[2])
# print(df['Name'][0:5])
# print(df.Name)
# print(df[['Name', 'HP']]) # for multiple columns

# print(df.iloc[1]) # -> index location for row
# print(df.iloc[1:4]) # -> index location
# print(df.iloc[2,1]) # -> index location

# for index, row in df.iterrows():
#     print(index, row['Name'])

# print(df.loc[df['Name'] == "Pikachu"]) #used to find specific data in dataset where data is more textual based rather than index based
# print(df.loc[df['Type 1'] == "Grass"])

# print(df.describe())
# print(df.sort_values('Name', ascending=False)) # -> sort data in descending order
# print(df.sort_values(['Type 1', 'HP'], ascending=[1, 0])) # -> sort data in ascending order for first column and descending order for second column

# df['Total'] = df['HP'] + df['Attack'] + df['Defense'] + df['Sp. Atk'] + df['Sp. Def'] + df['Speed']
# print(df.head(5))
# now delete the total column
# df = df.drop(columns=['Total'])
# print(df.head(5))

# df['Total'] = df.iloc[:, 4:10].sum(axis=1) # axis = 1 because to add horizontally
# print(df.head(5)) # -> print first 5 rows of the dataframe

# now to change the location of the total column
# cols = list(df.columns.values)
# df = df[cols[0:4] + [cols[-1]] + cols[4:12]] # doesnt modify data itself just visualize

# df.to_csv('modified.csv', index=False)
# df.to_excel('modified.xlsx', index=False)]
# df.to_csv('modified.txt', index=False, sep='\t')

# new_df = df.loc[(df['Type 1'] == 'Grass') & (df['Type 2'] == 'Poison') & (df['HP'] > 70)] # to have the dataframe which have only grass type pokemon
# new_df = new_df.reset_index() # -> to reset the index of the new dataframe}
# new_df.to_csv('filtered.csv') # -> to save the new dataframe to a csv file

# new_df = new_df.reset_index() # it adds the new column where index starts from 0
# new_df.reset_index(drop = True, inplace = True) # it drops the index column because from the last command the index column is added

# print(df.loc[~df['Name'].str.contains('Mega')])

import re
# print(df.loc[df['Type 1'].str.contains('fire|grass', flags=re.I, regex=True)]) # -> ignore case when we use flags=re.I
# print(df.loc[df['Name'].str.contains('^pi[a-z]*', flags=re.I, regex=True)]) # -> to find pokemon whose name starts with pi and [a-z] for next set of letters and * for zero or more
# and if we use ^ at the beginning then it will find pokemon whose name starts with pi

# df.loc[df['Type 1'] == 'Flamer', 'Type 1'] = 'Fire' # -> to change the type 1 of fire pokemon to flamer
# print(df)

# df.loc[df['Total'] > 500, ['Generation', 'Legendary']] = ['Test 1', 'Test 2'] # -> to change the generation and legendary column to test 1 and test 2
# print(df)

# aggregate functions
# Assuming df is your DataFrame
# Select only numeric columns before applying groupby and mean
# numeric_columns = df.select_dtypes(include='number').columns # important because we need to select only numeric columns
# result = df.groupby(['Type 1'])[numeric_columns].mean().sort_values('Attack', ascending=False)
# result = df.groupby(['Type 1'])[numeric_columns].sum()
# result = df.groupby(['Type 1'])[numeric_columns].count()
# print(result)

# df['count'] = 1
# print(df.groupby(['Type 1', 'Type 2']).count()['count'])

# working with large datasets
new_df = pd.DataFrame(columns = df.columns) # -> to create a new dataframe with the same columns as the original dataframe
for df in pd.read_csv('modified.csv', chunksize = 5): # 5 rows of modified csv
    result = df.groupby(['Type 1']).count()
    new_df = pd.concat([new_df, result])

print(new_df)