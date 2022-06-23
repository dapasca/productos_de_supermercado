# DATAMARKET VISUALIZATION 
## SNGULAR

https://datamarket.es
Productos de supermercados

Visualization about "productos de supermercado" based on data from supermarket data from [datamarket](https://datamarket.es/#productos-de-supermercados-dataset)

This repository only contains the EDA scripts.
Tableau visualization can be found in **Alfred Sngular** public profile

### How it works
Install requirements
```bash
pip install -r requirements.txt
```

Class `SupermarketCategories` from *supermarket_categories.py* allows to load the data and merge it with categories from *split_categories.csv* 
```python
import pandas as pd
from supermarket_categories import SupermarketCategories

df = pd.read_csv('datamarket_productos_de_supermercados.csv')
sc = SupermarketCategories(df)

sc.dataset.head()
```

### Notebooks
There is also notebooks available with preliminary analysis and EDA from the data source

Creators
https://github.com/dapasca
