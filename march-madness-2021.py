import numpy as mp
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor


def main():
    # read the csv results
    df2013 = pd.read_csv('cbb13.csv')
    df2014 = pd.read_csv('cbb14.csv')
    df2015 = pd.read_csv('cbb15.csv')
    df2016 = pd.read_csv('cbb16.csv')
    df2017 = pd.read_csv('cbb17.csv')
    df2018 = pd.read_csv('cbb18.csv')
    df2019 = pd.read_csv('cbb19.csv')
    df2021 = pd.read_csv('cbb21.csv')

    # drop the rows that have teams that were not seeded (should have 68 total teams per year)
    df2013 = df2013.dropna()
    df2014 = df2014.dropna()
    df2015 = df2015.dropna()
    df2016 = df2016.dropna()
    df2017 = df2017.dropna()
    df2018 = df2018.dropna()
    df2019 = df2019.dropna()
    df2021 = df2021.dropna()

    # confirm that there were 68 teams in the bracket each year
    print('Teams in 2013: {}'.format(len(df2013)))
    print('Teams in 2014: {}'.format(len(df2014)))
    print('Teams in 2015: {}'.format(len(df2015)))
    print('Teams in 2016: {}'.format(len(df2016)))
    print('Teams in 2017: {}'.format(len(df2017)))
    print('Teams in 2018: {}'.format(len(df2018)))
    print('Teams in 2019: {}'.format(len(df2019)))
    print('Teams in 2021: {}'.format(len(df2021)))

    df = df2013.append(df2014)
    df = df.append(df2015)
    df = df.append(df2016)
    df = df.append(df2017)
    df = df.append(df2018)
    df = df.append(df2019)

    # determine how many games each team won
    df = df.replace({'R68':0, 'R64':0, 'R32':1, 'S16':2, 'E8':3, 'F4':4, '2ND':5, 'Champions':6})
    df2021 = df2021.replace({'R68':0, 'R64':0, 'R32':1, 'S16':2, 'E8':3, 'F4':4, '2ND':5, 'Champions':6})

    # create DataFrames for the model inputs and outputs
    df_inputs = df.drop(['POSTSEASON'], axis=1)
    # need to drop the columns with text in them, because we cannot create regressions with non-numeric data
    df_inputs = df_inputs.drop(['TEAM'], axis=1)
    df2021_teams = df2021['TEAM']
    df2021 = df2021.drop(['TEAM'], axis=1)
    df_inputs = df_inputs.drop(['CONF'], axis=1)
    df2021 = df2021.drop(['CONF'], axis=1)
    df_outputs = df['POSTSEASON']

    # print some results to view
    plt.figure()
    plt.scatter(df['W'], df['POSTSEASON'])
    plt.xlabel('Number of games won (W)')
    plt.ylabel('Number of March-Madness wins')
    plt.savefig('Number-of-games-won')

    plt.figure()
    plt.scatter(df['BARTHAG'], df['POSTSEASON'])
    plt.xlabel('Barthag Parameter')
    plt.ylabel('Number of March-Madness wins')
    plt.savefig('Barthag-parameter')

    # make predictions using a linear regression model
    myLinearRegression = linear_model.LinearRegression()
    myLinearRegression.fit(df_inputs, df_outputs)
    myLinearPredictions = myLinearRegression.predict(df2021)
    myLinearImportance = myLinearRegression.coef_

    print()
    print('----------- Linear Regression -------------')
    print('Team : Prediction')
    print('-------------------------------------------')
    for iii in range(len(myLinearPredictions)):
        print('{} : {}'.format(df2021_teams[iii], myLinearPredictions[iii]))

    # print the coefficents
    print()
    print('Parameter : Score')
    print('-------------------------------------------')
    for iii in range(len(myLinearImportance)):
        print('{} \t: {}'.format(df_inputs.columns[iii], myLinearImportance[iii]))

    # plot the coefficients -> clearly BARTHAG is the most influential
    plt.figure()
    plt.bar([x for x in range(len(myLinearImportance))], myLinearImportance)
    plt.savefig('Linear-parameter-importance')

    plt.figure()
    plt.bar([x for x in range(len(myLinearPredictions))], myLinearPredictions)
    plt.savefig('Linear-team-predictions')

    # make predictions using the Lasso linear model
    myLassoRegression = linear_model.Lasso(alpha=0.001)
    myLassoRegression.fit(df_inputs, df_outputs)
    myLassoPredictions = myLassoRegression.predict(df2021)
    myLassoImportance = myLassoRegression.coef_

    # print results from Lasso linear regression
    print()
    print('--------- Lasso Linear Regression ---------')
    print('Team : Prediction')
    print('-------------------------------------------')
    for iii in range(len(myLassoPredictions)):
        print('{} : {}'.format(df2021_teams[iii], myLassoPredictions[iii]))
    print()
    print('Parameter : Score')
    print('-------------------------------------------')
    for iii in range(len(myLassoImportance)):
        print('{} \t: {}'.format(df_inputs.columns[iii], myLassoImportance[iii]))

    # plot the coefficients -> clearly BARTHAG is the most influential
    plt.figure()
    plt.bar([x for x in range(len(myLassoImportance))], myLassoImportance)
    plt.savefig('Lasso-parameter-importance')

    plt.figure()
    plt.bar([x for x in range(len(myLassoPredictions))], myLassoPredictions)
    plt.savefig('Lasso-team-predictions')

    # use a random forest tree to generate predictions
    randTree = RandomForestRegressor(min_samples_split=20, random_state=6)
    randTree.fit(df_inputs, df_outputs)
    randTreeImportance = randTree.feature_importances_
    randTreePredictions = randTree.predict(df2021)

    # print results from the random tree
    print()
    print('--------- Random Tree Regression ---------')
    print('Team : Prediction')
    print('-------------------------------------------')
    for iii in range(len(randTreePredictions)):
        print('{} : {}'.format(df2021_teams[iii], randTreePredictions[iii]))
    print()
    print('Parameter : Score')
    print('-------------------------------------------')
    for iii in range(len(randTreeImportance)):
        print('{} \t: {}'.format(df_inputs.columns[iii], randTreeImportance[iii]))

    plt.figure()
    plt.bar([x for x in range(len(randTreePredictions))], randTreePredictions)
    plt.savefig('Random-tree-team-predictions')

    plt.figure()
    plt.bar([x for x in range(len(randTreeImportance))], randTreeImportance)
    plt.savefig('Random-tree-parameter-importance')


if __name__ == '__main__':
    main()