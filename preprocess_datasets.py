import numpy as np
import pandas
from sklearn.preprocessing import normalize,LabelEncoder,OrdinalEncoder


def preprocess_index(PATH,PATH1,i):
    print('Reading feature info...')
    data_info = pandas.read_csv(PATH, encoding = "ISO-8859-1", header=None).values  
    # feature_names = data_info[:, 1]  
    feature_types = np.array([item.lower() for item in data_info[:, 2]])  
                         

    print('Finding column indices for feature types...')
    nominal_cols = np.where(feature_types == "nominal")[0]-1
    integer_cols = np.where(feature_types == "integer")[0]-1
    binary_cols = np.where(feature_types == "binary")[0]-1
    float_cols = np.where(feature_types == "float")[0]-1

    # nominal_names = feature_names[nominal_cols]
    # integer_names = feature_names[integer_cols]
    # binary_names = feature_names[binary_cols]
    # float_names = feature_names[float_cols]

    print('Reading csv files...')
    dataframe = pandas.read_csv(PATH1, header=None)

    print('Preprocessing...')
    print('Converting data...')

    dataframe[integer_cols] = dataframe[integer_cols].apply(pandas.to_numeric, errors='coerce')
    dataframe[binary_cols] = dataframe[binary_cols].apply(pandas.to_numeric, errors='coerce')
    dataframe[float_cols] = dataframe[float_cols].apply(pandas.to_numeric, errors='coerce')
    dataframe[nominal_cols] = dataframe[nominal_cols].astype(str)

    print('Replacing NaNs...')
    dataframe.loc[:,47] = dataframe.loc[:,47].replace('nan','normal', regex=True).apply(lambda x: x.strip().lower())
    dataframe.loc[:,binary_cols] = dataframe.loc[:,binary_cols].replace(np.nan, 0, regex=True)
    dataframe.loc[:,37:39] = dataframe.loc[:,37:39].replace(np.nan, 0, regex=True)
    dataframe.loc[:,float_cols] = dataframe.loc[:,float_cols].replace(np.nan, 0, regex=True)

    print('Stripping nominal columns and setting them lower case...')
    dataframe.loc[:,nominal_cols] = dataframe.loc[:,nominal_cols].applymap(lambda x: x.strip().lower())

    print('Changing targets \'backdoors\' to \'backdoor\'...')
    dataframe.loc[:,47] = dataframe.loc[:,47].replace('backdoors','backdoor', regex=True).apply(lambda x: x.strip().lower())
    dataset = dataframe.values

    print('Slicing dataset...')
    nominal_x = dataset[:, nominal_cols][:,:]
    integer_x = dataset[:, integer_cols][:,:].astype(np.float32)
    binary_x = dataset[:, binary_cols][:,:].astype(np.float32)
    float_x = dataset[:, float_cols][:,:].astype(np.float32)


    print(nominal_x.shape)
    nominal_x1=nominal_x[:,0]
    nominal_x2=nominal_x[:,1]
    nominal_x3=nominal_x[:,2]
    nominal_x4=nominal_x[:,3]
    nominal_x5=nominal_x[:,4]
    nominal_x6=nominal_x[:,5]
    print('Encoder nominal data...')
    v =LabelEncoder() 
    o = OrdinalEncoder()
#D = map(lambda dataline: dict(zip(nominal_names, dataline)), nominal_x)
    labeled_nominal_x1 = o.fit_transform(nominal_x1.reshape(-1,1)).astype(np.float32)
    labeled_nominal_x2 = o.fit_transform(nominal_x2.reshape(-1,1)).astype(np.float32) 
    labeled_nominal_x3 = o.fit_transform(nominal_x3.reshape(-1,1)).astype(np.float32) 
    labeled_nominal_x4 = o.fit_transform(nominal_x4.reshape(-1,1)).astype(np.float32) 
    labeled_nominal_x5 = o.fit_transform(nominal_x5.reshape(-1,1)).astype(np.float32) 
    labeled_nominal_x6 = v.fit_transform(nominal_x6).astype(np.float32) 
    print('Concatenating X...')
    normalized_X = np.column_stack((labeled_nominal_x6, labeled_nominal_x1, labeled_nominal_x2,
    labeled_nominal_x3,labeled_nominal_x4,labeled_nominal_x5,integer_x,float_x, binary_x))

    normalized_X=pandas.DataFrame(normalized_X)

    #去除nan
    print(np.any(np.isnan(normalized_X)))
    normalized_X=normalized_X.dropna(axis=0,how='any')
    print(np.any(np.isnan(normalized_X)))

    print('Save processed files')
    normalized_X.to_csv("./pre_datasets/pre_UNSW-NB15_"+i+".csv",mode='a',index=False,header=None,sep=",")


if __name__ == '__main__':

    for i in range(1,5):
        i=str(i)
        path = './raw_datasets/UNSW-NB15_features.csv'
        path1='./raw_datasets/UNSW-NB15_'+i+'.csv'
        preprocess_index(path,path1,i)