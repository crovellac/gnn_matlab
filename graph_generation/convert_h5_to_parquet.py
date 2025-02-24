import pandas as pd
df=pd.read_hdf('train.h5','table').sample(n=30000)
df.to_parquet('jets_train.parquet')
df=pd.read_hdf('test.h5','table').sample(n=30000)
df.to_parquet('jets_test.parquet')
df=pd.read_hdf('val.h5','table').sample(n=30000)
df.to_parquet('jets_val.parquet')
