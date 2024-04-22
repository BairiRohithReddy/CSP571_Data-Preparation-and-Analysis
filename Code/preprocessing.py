#defining an automaing function

def univariate(df):
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt, seaborn as sns
    
    output_df = pd.DataFrame(columns = ['type', 'count', 'unique', 'missing','mean','mode','Q1(25%)','median/Q2(50%)', 'Q3(75%)',
                                       'min','max', 'std', 'kurtosis', 'skew'])
    
    for col in df:
        #calculate metrics that apply to all dtypes
        dtype = df[col].dtype
        count = df[col].count()
        missing = df[col].isna().sum()
        unique = df[col].nunique()
        mode = df[col].mode()[0]
        
        if pd.api.types.is_numeric_dtype(df[col]):
            
            #calculate metrics that only apply for numeric types
            min_v = df[col].min()
            max_v = df[col].max()
            mean = df[col].mean()
            Q1 = df[col].quantile(0.25)
            median = df[col].quantile(0.50)
            Q3 = df[col].quantile(0.75)
            std = df[col].std()
            kurt = df[col].kurt()
            skew = df[col].skew()
            
            
            output_df.loc[col] = [dtype, count, unique, missing,mean,mode,Q1,median,Q3,min_v,max_v,std,kurt,skew]
            
            sns.histplot(data=df,x=col)
            
            
        else:
            output_df.loc[col] = [dtype, count, unique, missing,'-',mode,'-','-','-','-','-','-','-','-']
            sns.countplot(data=df,x=col)
        
        plt.show()
        
    return output_df.transpose()

#Now moving onto bivariate analysis
# defining a function for bivariate analysis

# defining a function for bivariate analysis

def bivariate(df, label, roundto=4):
    import pandas as pd
    from scipy import stats

    output_df = pd.DataFrame(
        columns=['missing %', 'skew', 'type', 'unique', 'p', 'r', 'τ', 'ρ', 'y = m(x) + b', 'F', 'X2'])

    # col here represents the individual features in the dataframe
    for col in df:

        if col != label:

            # calculate stats that apply for all datatypes
            df_temp = df[[col, label]].copy()
            df_temp = df_temp.dropna().copy()
            missing = ((df.shape[0] - df_temp.shape[0]) / df.shape[0]) * 100
            dtype = df_temp[col].dtype
            unique = df_temp[col].nunique()

            if pd.api.types.is_numeric_dtype(df_temp[col]) and pd.api.types.is_numeric_dtype(df_temp[label]):
                # working with N-N relationships
                # print(col, stats.linregress(df[col],df[label]))
                m, b, r, p, e = stats.linregress(df_temp[col], df_temp[label])
                skew = df_temp[col].skew()
                tau, tp = stats.kendalltau(df_temp[col], df_temp[label])
                rho, rp = stats.spearmanr(df_temp[col], df_temp[label])
                output_df.loc[col] = [f'{round(missing, roundto)}%', skew, dtype, unique, round(p, roundto),
                                      round(tau, roundto), round(rho, roundto), round(r, roundto),
                                      f'y = {round(m, roundto)}*x+{round(b, roundto)}', '-', '-']

                # visualizing the N2N relationships
                scatterplot(df_temp, col, label)

            elif not pd.api.types.is_numeric_dtype(df_temp[col]) and not pd.api.types.is_numeric_dtype(df_temp[label]):
                # working with C-C relationships

                contingency_table = pd.crosstab(df_temp[col], df_temp[label])
                X2, p, dof, expected = stats.chi2_contingency(contingency_table)

                output_df.loc[col] = [f'{round(missing, roundto)}%', '-', dtype, unique, round(p, roundto), '-', '-',
                                      '-', '-', '-', round(X2, roundto)]

                # now visualizing the C2C relationships
                crosstab(df_temp, feature, label)

            else:
                # N-C relationships or C-N relationships
                if pd.api.types.is_numeric_dtype(df_temp[col]):
                    skew = round(df_temp[col].skew(), roundto)
                    num = col
                    cat = label

                else:
                    skew = '-'
                    num = label
                    cat = col

                groups = df_temp[cat].unique()
                group_list = []
                for g in groups:
                    group_list.append(df_temp[df_temp[cat] == g][num])

                f, p = stats.f_oneway(*group_list)  # <-same as (group_list[0], group_list[1],.....group_list[n])

                output_df.loc[col] = [f'{round(missing, roundto)}%', skew, dtype, unique, round(p, roundto), '-', '-',
                                      '-', '-', round(f, roundto), '-']

                # visualizing the C2N relationships
                barchart(df_temp, col, num)

    return output_df  # .sort_values(by = 'p', ascending = True)

# Automating Bivariate Relationships - Visualizations

# This scatterplot will help us visualize the N2N relations
def scatterplot(df, feature, label, roundto=4, linecolor='Red'):
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    from scipy import stats

    # creating the plot
    # line_kws is used to pass parameters to improve the trendline and make it look better
    # regplot plots the trendline along with the scatterplot
    sns.regplot(x=df[feature], y=df[label], line_kws={'color': linecolor})

    # adding regression line to the scatterplot
    m, b, r, p, err = stats.linregress(df[feature], df[label])
    tau, tp = stats.kendalltau(df[feature], df[label])
    rho, rp = stats.spearmanr(df[feature], df[label])
    fskew = round(df[feature].skew(), roundto)
    lskew = round(df[label].skew(), roundto)

    # Adding all those values to the plot
    textstr = f'y = {round(m, roundto)}x + {round(b, roundto)}\n'
    textstr += f'r = {round(r, roundto)}, p = {round(p, roundto)}\n'
    textstr += f'tau = {round(tau, roundto)}, p = {round(tp, roundto)}\n'
    textstr += f'rho = {round(rho, roundto)}, p = {round(rp, roundto)}\n'
    textstr += f'{feature} skew = {round(df[feature].skew(), roundto)}\n'
    textstr += f'{label} skew = {round(df[label].skew(), roundto)}'

    # adding the correlation matrix
    # df.corr

    plt.text(0.95, 0.2, textstr, fontsize=12, transform=plt.gcf().transFigure)
    plt.show()


# Now we shall make bar chart to visualize N2C/ C2N

def barchart(df, feature, label, roundto=4, p_threshold=0.05, sig_ttest=True):
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    from scipy import stats

    # making sure that feature is categorical and label is numeric
    if pd.api.types.is_numeric_dtype(df_temp[feature]):
        num = feature
        cat = label
    else:
        num = label
        cat = feature

    # creating the bar chart
    sns.barplot(x=df[cat], y=df[num])

    # creating the numeric lists needed to calculate the ANOVA
    groups = df[feature].unique()
    group_list = []
    for g in groups:
        group_list.append(df[df[cat] == g][num])

    f, p = stats.f_oneway(*group_list)  # <-same as (group_list[0], group_list[1],.....group_list[n])

    # calculate individual pairwise t-test for each pair of columns
    ttests = []
    for i1, g1 enumerate(groups):
        for i2, g2 enumerate(groups):
            if i1 < i2:
                list1 = df[df[cat] == g1][num]
                list2 = df[df[cat] == g2][num]
                t, tp = stats.ttest_ind(list1, list2)
                ttests.append([f'{g1} - {g2}', round(t, roundto), round(tp, roundto)])
                # print(i1, g1, i2, g2)
    print(ttests)

    # make a bonferroni correction --> adjust the p-value threshold to be 0.05 / n of ttests
    # in simple terms we just need a smaller p-value to call it significant
    # so instead of calculating p-values for all the columns and printing them out we can just print the significant ones by filtering them out
    bonferroni = p_threshold / len(ttests)

    # creating textstr to add text to our chart
    textstr = f'F: {round(f, roundto)}\n'
    textstr += f'p: {round(p, roundto)}'
    textstr += f'Bonferroni p:  {bonferroni}'
    for ttest in ttests:
        if sig_ttest:
            if ttests[2] <= bonferroni:
                textstr += f'\n {ttests[0]}: t : {ttest[1]}, p: {ttest[2]}'
            else:
                textstr += f'\n {ttests[0]}: t : {ttest[1]}, p: {ttest[2]}'

                # if there are too many feature gorups, print x labels vertically
    if df[feature].nunique() > 7:
        plt.xticks(rotation=90)

    plt.text(0.95, 0.10, textstr, fontsize=12, transform=plt.gcf().transFigure)
    # plt.xticks(rotation = 90)
    plt.show()


# creating a crosstab to visualize C2C relationships

def crosstab(df, feature, label, roundto=4):
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from scipy import stats
    import numpy as np

    # generating the crosstab
    crosstab = pd.crosstab(df[feature], df[label])

    # calculate the statistics
    X, p, dof, contingency_table = stats.chi2_contingency(crosstab)

    # creating a heatmap
    # sns.heatmap(contingency_table)

    # creating the text string
    textstr = f'X2: {round(X, roundto)}\n'
    textstr += f'p: {round(p, roundto)}'

    # creating the crosstab dataframe ,heatmap and passing the values and the column names inside the heatmap
    # now with this we get our column names on top of the graph along with the stats and heatmap
    ct_df = pd.DataFrame(np.rint(contingency_table).astype('int64'), columns=crosstab.columns, index=crosstab.index)
    sns.heatmap(ct_df, annot=True, fmt='d', cmap='Paired')

    plt.text(0.95, 0.2, textstr, fontsize=12, transform=plt.gcf().transFigure)
    plt.show()


def basic_wrangling(df, features=[], missing_threshold=0.95, unique_threshold=0.95, messages=True):
    import pandas as pd

    if len(features) == 0: features = df.columns
    for feature in features:
        if feature in df.columns:
            missing = df[feature].isna().sum()
            unique = df[feature].nunique()
            rows = df.shape[0]

            if missing / rows >= missing_threshold:
                if messages: print(
                    f"Too much missing ({missing} out of {rows}, {(round(missing / rows, 0))}) for {feature}")
                df.drop(columns=[feature], inplace=True)

            elif unique / rows >= unique_threshold:
                if df[feature].dtype in ['int64', 'object']:
                    if messages: print(
                        f"Too many unique values ({unique} out of {rows}, {round(unique / rows, 0)}) for {feature}")
                    df.drop(columns=[feature], inplace=True)

            elif unique == 1:
                if messages: print(f"only one value({df.feature.unique()[0]}) for {feature}")
                df.drop(columns=[feature], inplace=True)
        else:
            if messages: print(f"The feature \"{feature}\" doesn't exist as spelled in the dataframe provided")

    return df

#Managing Date and Time

def parse_date(df, features=[], drop_date=True, messages=True):
    import pandas as pd

    for feature in features:
        if feature in df.columns:
            df[feature] = pd.to_datetime(df[feature])
            df[f'{feature}_year'] = df[feature].dt.year
            df[f'{feature}_month'] = df[feature].dt.month
            df[f'{feature}_day'] = df[feature].dt.day
            df[f'{feature}_weekday'] = df[feature].dt.day_name()

            if drop_date: df.drop(columns=[feature], inplace=True)
        else:
            if messages: print(f'{feature} does not exist as spelled in the dataframe provided')
    return df

#Bin low count group values

def bin_categories(df, features=[], cutoff=0.015, replace_with='Other', messages=True):
    import pandas as pd

    for feature in features:
        if feature in df.columns:
            if not pd.api.types.is_numeric_dtype(df[feature]):
                other_list = df[feature].value_counts()[df[feature].value_counts() / df.shape[0] < cutoff].index
                df.loc[df[feature].isin(other_list), feature] = replace_with
        else:
            if messages: print(f'{feature} not found in the dataframe provided as spelled, No binning performed')
    return df

