import numpy as np
import pandas as pd
df = pd.read_csv('week2.csv')
print (df . head ( ) )
X1=df . iloc [ : , 0 ]
X2=df . iloc [ : , 1 ]
X=np . column_stack ( ( X1, X2 ) )
y=df . iloc [:, 2]