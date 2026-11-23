import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder , OrdinalEncoder , StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import KNNImputer


class handling_data:
    def __init__(self):
        self.df = None
        self.before_model = None
        self.knn_target = None

        self.group_model = None

        self.map_model = None

        self.group_year = None
        self.mapping_year = None

        self.group_fuel = None
        self.mapping_fuel = None

        self.group_engine = None
        self.mapping_engine=None

        self.group_trans = None
        self.map_trans = None

        self.mapping_tax = None

        self.mapping_mileage = None

        self.mapping_mpg = None

        self.target_encode = None

        self.bounds = {}

    def handle_data(self , df):
        df = df.drop_duplicates()

        for col in df.select_dtypes(include='object').columns:
            df[col] = df[col].str.extract(r'([A-Za-z0-9]+)')
            df[col] = df[col].str.capitalize()
        
        return df


    def handle_make(self , df):
        df['Make'] = df.groupby('model')['Make'].ffill().bfill()

        return df


    def handle_model(self , df):
        self.group_model = (df.groupby(['transmission', 'Make' ,'fuelType'])[['model']]
            .value_counts(dropna=False)
            .to_frame()
            .reset_index(level=[0,1,2])
            .groupby(['transmission' , 'Make' , 'fuelType'])['count']
            .idxmax()
            .reset_index()
        )


        for trans in df['transmission'].unique():
            for make in df['Make'].unique():
                for fuel in df['fuelType'].unique():
                    
                    mask_group = (self.group_model['transmission'] == trans) & (self.group_model['Make'] == make) & (self.group_model['fuelType'] == fuel)
                    fill_array = self.group_model[mask_group]['count']

                    if fill_array.shape[0] > 0 :
                        fill_= fill_array.iloc[0]

                        mask_df = ((df['transmission']==trans) & (df['Make'] == make) & (df['fuelType'] == fuel))
                        df.loc[mask_df , 'model'] = df.loc[mask_df , 'model'].fillna(fill_)

        
        self.map_model = self.group_model.set_index(['transmission' , 'Make' , 'fuelType'])['count'].to_dict()
        df.loc[:, 'model'] = df['model'].fillna(
                pd.Series(
                    df.set_index(['transmission','Make','fuelType']).index.map(self.map_model)
                )
            )
        
        df.loc[:, 'model'] = df['model'].fillna(
                df['model'].mode()[0]
            )
        
        return df
    
        
    def handle_year(self , df):
        self.group_year = (df.groupby(['Make' , 'model'])['year']
            .value_counts()
            .to_frame()
            .reset_index(level=[0,1])
            .groupby(['Make','model'])['count']
            .idxmax()
            .to_frame()
            .reset_index()
            .rename(columns={'count' : 'year'}))

        
        for make in df['Make'].unique():
            for model in df['model'].unique():
                
                mask_group = (self.group_year['Make'] == make) & (self.group_year['model'] == model)
                fill_array = self.group_year[mask_group]['year']

                if fill_array.shape[0] > 0 :
                    fill_= fill_array.iloc[0]

                    mask_df = ((df['Make']==make) & (df['model'] == model))
                    df.loc[mask_df , 'year'] = df.loc[mask_df , 'year'].fillna(fill_)


        self.mapping_year = self.group_year.set_index(['Make' , 'model'])['year'].to_dict()

        df.loc[:, 'year'] = df['year'].fillna(
                pd.Series(
                    df.set_index(['Make' , 'model']).index.map(self.mapping_year)
                )
            )


        df.loc[:, 'year'] = df['year'].fillna(
            df['year'].mode()[0]
        )

        return df
    
        
    def handle_fueltype(self , df):
        self.group_fuel = (df.groupby('model')['fuelType']
                .value_counts()
                .to_frame()
                .reset_index('model')
                .groupby(['model'])['count']
                .idxmax()
                .to_frame()
                .reset_index()
        )

        for model in df['model'].unique():
    
            mask_group = (self.group_fuel['model'] == model)
            fill_array = self.group_fuel[mask_group]['count']

            if fill_array.shape[0] > 0 :
                fill_= fill_array.iloc[0]

                mask_df =(df['model'] == model)& (df['fuelType'].isna())
                df.loc[mask_df , 'fuelType'] = df.loc[mask_df , 'fuelType'].fillna(fill_)
        

        self.mapping_fuel = self.group_fuel.set_index('model')['count'].to_dict()

        df.loc[: , 'fuelType'] = df['fuelType'].fillna(
                    df['model'].map(self.mapping_fuel)
            )
        

        df.loc[:, 'fuelType'] = df['fuelType'].fillna(
            df['fuelType'].mode()[0]
        )

        
        return df

    
    def handle_enginesize(self , df):
        self.group_engine = df.groupby(['model' , 'fuelType'])['engineSize'].value_counts().to_frame()

        self.group_engine = self.group_engine.reset_index(level=[0,1]).groupby(['model' , 'fuelType'])['count'].idxmax().to_frame().reset_index()

        for model in df['model'].unique():
            for fuel in df['fuelType'].unique():
                mask_group = (self.group_engine['fuelType'] == fuel) & (self.group_engine['model'] == model)
                fuel_array = self.group_engine[mask_group ]['count']

                if fuel_array.shape[0] > 0 :
                    fill_ = fuel_array.iloc[0]

                    mask_df = (df['model'] == model) & (df['fuelType'] == fuel) & (df['engineSize'].isna())
                    df.loc[mask_df , 'engineSize'] = df.loc[mask_df , 'engineSize'].fillna(fill_)


        self.mapping_engine = self.group_engine.set_index(['model','fuelType'])['count'].to_dict()

        df['engineSize'] = df['engineSize'].fillna(
                pd.Series(
                    df.set_index(['model' , 'fuelType']).index.map(self.mapping_engine)
                )
            )
        

        df.loc[:, 'engineSize'] = df['engineSize'].fillna(
            df['engineSize'].mode()[0]
        )

        return df
    

        
    def handle_transmation(self,df):
        self.group_trans = df.groupby('model')['transmission'].value_counts().to_frame().reset_index(['model']).groupby('model').idxmax().reset_index()

        for model in df['model'].unique():
            mask = (df['model'] == model) & (df['transmission'].isna())

            fill_array = self.group_trans[self.group_trans['model']==model]

            if fill_array.shape[0] > 0:
                fill_ = fill_array['count'].iloc[0]
                df.loc[mask , 'transmission'] = fill_

        
        self.map_trans = df.groupby('model')['transmission'].value_counts().to_frame().reset_index(['model']).groupby('model').idxmax()['count'].to_dict()

        df.loc[: ,'transmission'] = df['transmission'].fillna(
                pd.Series(
                    df.set_index('model').index.map(self.map_trans)
                )
            )
        

        df.loc[:, 'engineSize'] = df['engineSize'].fillna(
            df['engineSize'].mode()[0]
        )
        

        
        return df

        
    def handle_tax_mileage_mpg(self,df):
        self.mapping_tax = df.groupby('Make')['tax'].mean().to_dict()
        df.loc[: ,'tax'] = df['tax'].fillna(df['Make'].map(self.mapping_tax))


        self.mapping_mileage = df.groupby('Make')['mileage'].median().to_dict()
        df.loc[: ,'mileage'] = df['mileage'].fillna(df['Make'].map(self.mapping_mileage))


        self.mapping_mpg = df.groupby('Make')['mpg'].mean().to_dict()
        df.loc[: ,'mpg'] = df['mpg'].fillna(df['Make'].map(self.mapping_mpg))

        return df


    def target_endode(self , df):
        self.target_encode = df.groupby('Make')['price'].mean().to_dict()
        df['Make'] = df['Make'].map(self.target_encode)

        return df
    
    def scale_encode_data(self,df):
        preproccesor_linear = ColumnTransformer(
                transformers=[
                    ('num' ,  StandardScaler() , ['year' , 'mileage' , 'tax' , 'mpg' , 'engineSize']),
                    ('cat_OH' , OneHotEncoder(sparse_output=False , handle_unknown="ignore") , ['fuelType' , 'transmission']),
                    ('cat_ordinal' , OrdinalEncoder() , ['model'])
                ],remainder='passthrough'
            )

        self.before_model = Pipeline(
                steps=[
                    ('preproccesor' , preproccesor_linear)
                ]
            )


        df_transform = self.before_model.fit_transform(df)

        df_transform_df = pd.DataFrame(df_transform , columns=self.before_model.get_feature_names_out())

        return df_transform_df


    


    def handle_price(self , df_transform_df):
        
        self.knn_target = KNNImputer(n_neighbors=2)

        df_imputed = self.knn_target.fit_transform(df_transform_df)
        self.df = pd.DataFrame(df_imputed , columns = self.knn_target.get_feature_names_out())

        return self.df , self.before_model , self.knn_target
    

    def fit_transform(self , df):


        print(f'start processing ....')
        print(f'start shape : {df.shape}')

        df = self.handle_data(df)
        print(f'shape after initial handling : {df.shape}')


        df = self.handle_make(df)

        df = self.handle_model(df)
        print(f'shape after handle model : {df.shape}')

        df = self.handle_year(df)
        df = self.handle_fueltype(df)
        df = self.handle_enginesize(df)

        print(f'shape after handle enginesize : {df.shape}')

        df = self.handle_transmation(df)

        df = self.handle_tax_mileage_mpg(df)

        df = self.target_endode(df)

        df = self.scale_encode_data(df)




        df, before_model , knn = self.handle_price(df)

        print(f'the final shape : {df.shape}')

        return df , before_model , knn





    def transform(self , df):
        if self.df is None:
            raise Exception('Model not trained yet')
        
        ## make & prepare data

        df = self.handle_data(df)
        df = self.handle_make(df)


        ## model

        for trans in df['transmission'].unique():
            for make in df['Make'].unique():
                for fuel in df['fuelType'].unique():
                    
                    mask_group = (self.group_model['transmission'] == trans) & (self.group_model['Make'] == make) & (self.group_model['fuelType'] == fuel)
                    fill_array = self.group_model[mask_group]['count']

                    if fill_array.shape[0] > 0 :
                        fill_= fill_array.iloc[0]

                        mask_df = ((df['transmission']==trans) & (df['Make'] == make) & (df['fuelType'] == fuel))
                        df.loc[mask_df , 'model'] = df.loc[mask_df , 'model'].fillna(fill_)

        
        self.map_model = self.group_model.set_index(['transmission' , 'Make' , 'fuelType'])['count'].to_dict()
        df.loc[:, 'model'] = df['model'].fillna(
                pd.Series(
                    df.set_index(['transmission','Make','fuelType']).index.map(self.map_model)
                )
            )
        
        df.loc[:, 'model'] = df['model'].fillna(
                df['model'].mode()[0]
            )
        

        ## year
        
        for make in df['Make'].unique():
            for model in df['model'].unique():
                
                mask_group = (self.group_year['Make'] == make) & (self.group_year['model'] == model)
                fill_array = self.group_year[mask_group]['year']

                if fill_array.shape[0] > 0 :
                    fill_= fill_array.iloc[0]

                    mask_df = ((df['Make']==make) & (df['model'] == model))
                    df.loc[mask_df , 'year'] = df.loc[mask_df , 'year'].fillna(fill_)
                    


        df.loc[:, 'year'] = df['year'].fillna(
                pd.Series(
                    df.set_index(['Make' , 'model']).index.map(self.mapping_year)
                )
            )


        df.loc[:, 'year'] = df['year'].fillna(
            df['year'].mode()[0]
        )

    

        ## fueltype

        for model in df['model'].unique():
    
            mask_group = (self.group_fuel['model'] == model)
            fill_array = self.group_fuel[mask_group]['count']

            if fill_array.shape[0] > 0 :
                fill_= fill_array.iloc[0]

                mask_df =(df['model'] == model)& (df['fuelType'].isna())
                df.loc[mask_df , 'fuelType'] = df.loc[mask_df , 'fuelType'].fillna(fill_)
                

        

        df.loc[: , 'fuelType'] = df['fuelType'].fillna(
                    df['model'].map(self.mapping_fuel)
            )
        

        df.loc[:, 'fuelType'] = df['fuelType'].fillna(
            df['fuelType'].mode()[0]
        )


        ## enginesize

        for model in df['model'].unique():
            for fuel in df['fuelType'].unique():
                mask_group = (self.group_engine['fuelType'] == fuel) & (self.group_engine['model'] == model)
                fuel_array = self.group_engine[mask_group ]['count']

                if fuel_array.shape[0] > 0 :
                    fill_ = fuel_array.iloc[0]

                    mask_df = (df['model'] == model) & (df['fuelType'] == fuel) & (df['engineSize'].isna())
                    df.loc[mask_df , 'engineSize'] = df.loc[mask_df , 'engineSize'].fillna(fill_)
                    


        df['engineSize'] = df['engineSize'].fillna(
                pd.Series(
                    df.set_index(['model' , 'fuelType']).index.map(self.mapping_engine)
                )
            )
        

        df.loc[:, 'engineSize'] = df['engineSize'].fillna(
            df['engineSize'].mode()[0]
        )




        ## transmission

        for model in df['model'].unique():
            mask = (df['model'] == model) & (df['transmission'].isna())

            fill_array = self.group_trans[self.group_trans['model']==model]

            if fill_array.shape[0] > 0:
                fill_ = fill_array['count'].iloc[0]
                df.loc[mask , 'transmission'] = fill_

        
        df.loc[: ,'transmission'] = df['transmission'].fillna(
                pd.Series(
                    df.set_index('model').index.map(self.map_trans)
                )
            )
        
        
        df.loc[:, 'engineSize'] = df['engineSize'].fillna(
            df['engineSize'].mode()[0]
        )

        
        ## tax
        df.loc[: ,'tax'] = df['tax'].fillna(df['Make'].map(self.mapping_tax))

        ## mileage
        df.loc[: ,'mileage'] = df['mileage'].fillna(df['Make'].map(self.mapping_mileage))

        ## mpg
        df.loc[: ,'mpg'] = df['mpg'].fillna(df['Make'].map(self.mapping_mpg))

        


        df['Make'] = df['Make'].map(self.target_encode)

        
        


        df = self.before_model.transform(df)
        
        df = pd.DataFrame(df , columns=self.before_model.get_feature_names_out())



        df_imputed = self.knn_target.transform(df)
        df = pd.DataFrame(df_imputed , columns = self.knn_target.get_feature_names_out())

        return df





    
    def fit_detect_outliers(self, df):
        outlier_summary = {}

        for col in df.columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1

            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR

            self.bounds[col] = (lower, upper)

            outliers = df[(df[col] < lower) | (df[col] > upper)][col]

            if len(outliers) > 0:
                outlier_summary[col] = {
                    "count": len(outliers),
                    "lower_bound": lower,
                    "upper_bound": upper
                }

        return outlier_summary

    def view_outliers_with_values(self, df):
        if not self.bounds:
            raise Exception("Fit detect data first")

        outlier_data = {}

        for col in df.columns:
            lower, upper = self.bounds[col]

            outliers = df[(df[col] < lower) | (df[col] > upper)]

            if not outliers.empty:
                outlier_data[col] = outliers[col]

        return outlier_data


    def remove_outliers_with_values(self, df, exclude_cols=None):

        if not self.bounds:
            raise Exception("Fit detect data first")

        if exclude_cols is None:
            exclude_cols = []

        df_no_outliers = df.copy()

        for col in df.columns:

            if col in exclude_cols:
                continue

            lower, upper = self.bounds[col]

            df_no_outliers = df_no_outliers[
                (df_no_outliers[col] >= lower) &
                (df_no_outliers[col] <= upper)
            ]

        return df_no_outliers
    


        


        
