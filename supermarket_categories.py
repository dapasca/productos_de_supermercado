from typing import List

import numpy as np
import pandas as pd

from categories_dict import category_corrections, supercategories


class SupermarketCategories:
    def __init__(self,
                 dataset: pd.DataFrame,
                 load_split_categories: bool = True,
                 simplify_data: bool = True,
                 price_diff: bool = True,
                 remove_wrong_extreme_prices: bool = True,
                 full_series_boolean: bool = True):
        """
        Operate with "Productos de supermercados" dataset
        from https://datamarket.es

        :param dataset: the dataset from https://datamarket.es
        :param load_split_categories: If True, loads 'split_categories.csv'
        :param price_diff: If True, adds price diff to max date entries
        otherwise estimated if from self.dataset
        :param remove_wrong_extreme_prices: Remove wroung 'price' and
        'reference_price' values from Carrefour and Mercadona
        """

        self.dataset = dataset
        self.dataset['insert_date'] = \
            pd.to_datetime(self.dataset['insert_date']).dt.to_period('M')
        self.correct_categories()

        if load_split_categories:
            self.category_and_subcategories = \
                pd.read_csv('split_categories.csv')
        else:
            self.category_and_subcategories = self.iterative_split()
            self.category_and_subcategories.to_csv(
                'split_categories.csv', index=False)

        self.dataset = pd.merge(
            self.dataset, self.category_and_subcategories,
            on='category')

        if simplify_data:
            self.__simplify_dataset()
        if price_diff:
            self.__add_price_diff()
        if remove_wrong_extreme_prices:
            self.__remove_wrong_extreme_prices()
        if full_series_boolean:
            self.__full_series_boolean()

    def __full_series_boolean(self) -> pd.DataFrame:
        value_counts = self.dataset.product_id.value_counts()
        ids_full_series = value_counts[
            value_counts == value_counts.max()].index.values
        self.dataset['full_series'] = \
            self.dataset.product_id.isin(ids_full_series)

        return self.category_and_subcategories

    def correct_categories(self) -> pd.DataFrame:
        """
        Removes double '_' and
        'el_mercado' from categories and

        :return:
        """

        self.dataset.category = self.dataset.category.str.replace('__', '_')

        f_el_mercado = self.dataset.category.str[:10] == 'el_mercado'
        self.dataset.loc[f_el_mercado, 'category'] = \
            self.dataset.loc[f_el_mercado, 'category'].str[11:]
        return self.dataset

    @staticmethod
    def __split_with_junctions(category: str) -> List[str]:
        """
        Split a string by the '_' but
        ignoring the '_' surrounding junctions

        :param category: The string to be split
        :return: Split string
        """
        junction_list = ['y', 'e', 'de', 'del', 'para',
                         'el', 'la', 'en', 'con', 'sin',
                         'productos', 'cuidado']

        for junction in junction_list:
            category = \
                ('_' + category).replace(f'_{junction}_', f' {junction} ')[1:]
            category = \
                ('_' + category).replace(f' {junction}_', f' {junction} ')[1:]
            category = \
                ('_' + category).replace(f'_{junction} ', f' {junction} ')[1:]
        category_with_junctions = \
            [x.replace(' ', '_') for x in category.split('_')]
        return category_with_junctions

    def __split_category(self,
                         categories: List[str],
                         make_left_chunk: bool = True) -> pd.DataFrame:
        """
        Iteratively split categories in subcategories
        looking for the common substrings with other categories
        or in the same category

        :param categories: List of categories to analyze
        :param make_left_chunk: Split the category
        leaving the longest part at left
        :return: Dataframe with the category and the splits
        """

        category_list = np.unique(categories)
        category_list.sort()

        left_list = []
        right_list = []

        for i, cl in enumerate(category_list):

            left_category = None
            right_category = None

            category_after = self.__split_with_junctions(category_list[i - 1])
            category = self.__split_with_junctions(category_list[i])
            category_before = \
                self.__split_with_junctions(
                    category_list[i + 1]) if len(category_list) > i + 1 else []

            # Categories with repeated text split by repeated pattern
            if len(category) > 1:
                for w_i in range(1, (len(category) // 2) + 1):
                    if all([category[i] == category[i + w_i]
                            for i in range(w_i)]):
                        left_category = '_'.join(category[w_i:])
                        right_category = '_'.join(category[:w_i])
                        break

            # Categories split by comparing with previous and next category
            if left_category is None:

                match_after_i = len(category)
                match_before_i = len(category)
                for w_i, word in enumerate(category):

                    # Previous and next full_category corresponding word
                    if len(category_after) > w_i:
                        if word == category_after[w_i]:
                            match_after_i = w_i
                    if len(category_before) > w_i:
                        if word == category_before[w_i]:
                            match_before_i = w_i

                    # append to left_list and right_list
                    if make_left_chunk:
                        similarity_condition = \
                            match_after_i == w_i or match_before_i == w_i
                    else:
                        similarity_condition = \
                            (match_after_i == w_i and match_before_i >= w_i) \
                            or \
                            (match_before_i == w_i and match_after_i >= w_i)

                    if similarity_condition:
                        if w_i + 1 < len(category):
                            pass
                        else:
                            left_category = '_'.join(category)
                            right_category = ''
                    elif w_i > 0:
                        left_category = '_'.join(category[:w_i])
                        right_category = '_'.join(category[w_i:])
                        break
                    else:
                        left_category = '_'.join(category)
                        right_category = ''
                        break

            # Append the split category
            left_list.append(left_category)
            right_list.append(right_category)

        df_category = pd.DataFrame({'full_name': category_list,
                                    'first_category': left_list,
                                    'last_category': right_list})

        return df_category

    def iterative_split(self, save: bool = True) -> pd.DataFrame:
        """
        Apply iteratively self.__split_category until
        the deepest level or recursivity necessary to
        split the category in subcategories

        :param save: Save to file 'split_categories.csv'
        :return: Table with the categories and the split in subcategories
        """

        prefix = ''
        first_split = self.__split_category(self.dataset['category'].values)

        # SPLIT ITERATIVELY THE CATEGORY GENERATION UNNECESSARY COLUMNS
        # TODO: This only works with this data and 3 loops
        merge_with_previous_split = None
        for i in range(3):
            prefix = prefix + 'prev_'
            iter_split = self.__split_category(
                first_split['first_category'].values,
                make_left_chunk=False
            )
            merge_with_previous_split = pd.merge(first_split, iter_split,
                                                 left_on='first_category',
                                                 right_on='full_name',
                                                 suffixes=('_old', '_new'))
            merge_with_previous_split = merge_with_previous_split.drop(
                columns=['first_category_old', 'full_name_new'])
            merge_with_previous_split = merge_with_previous_split.rename(
                columns={
                    'full_name_old': 'full_name',
                    'last_category_old': 'last_category',
                    'first_category_new': 'first_category',
                    'last_category_new': f'{prefix}category',
                })

        merge_with_previous_split = \
            merge_with_previous_split.replace('', np.nan)
        merge_with_previous_split = merge_with_previous_split[
            ['full_name', 'first_category',
             'prev_prev_prev_category', 'last_category']]

        # CLEAN UNNECESSARY COLUMNS AND FORMAT THE DATAFRAME
        product_category = merge_with_previous_split['first_category']
        product_category.name = 'product_category'

        product_subcategory = \
            merge_with_previous_split.iloc[:, 2:].bfill(axis=1).iloc[:, 0]
        product_subcategory = product_subcategory.fillna('')
        product_subcategory.name = 'product_subcategory'

        # TODO: Recursive replace is not safe
        product_type = merge_with_previous_split.iloc[:, 2:].fillna('').apply(
            lambda x: '_'.join(x), axis=1).str.replace('__', '_').str.replace(
            '__', '_').str.strip('_')
        product_type.name = 'product_type'

        # Concat everything to generate the final table
        category_and_subcategories = pd.concat(
            [merge_with_previous_split.full_name,
             product_category,
             product_subcategory,
             product_type], axis=1)
        category_and_subcategories.product_type = \
            category_and_subcategories.apply(
                lambda x: x['product_type'][len(x['product_subcategory']):],
                axis=1).str.strip('_')

        category_and_subcategories = category_and_subcategories.rename(
            columns={'full_name': 'category',
                     'product_category': 'main_category',
                     'product_subcategory': 'secondary_category',
                     'product_type': 'type'}
        )

        # Replicate "secondary_category" in the empty "type" rows
        f_no_type = category_and_subcategories['type'] == ''
        category_and_subcategories.loc[f_no_type, 'type'] = \
            category_and_subcategories.loc[f_no_type, 'secondary_category']
        self.category_and_subcategories = category_and_subcategories

        # TODO: This correction is due to a mistake that must be solved
        self.__flip_wrong_columns()
        self.__handmade_category_corrections()
        self.__create_super_categories()

        if save:
            self.category_and_subcategories.to_csv('split_categories.csv',
                                                   index=False)
        return self.category_and_subcategories

    def __flip_wrong_columns(self) -> pd.DataFrame:
        """
        If columns are flipped in 'category_and_subcategories', correct it
        :return:
        """
        f_flipped = self.category_and_subcategories.apply(
            lambda x: x['category'].find(x['secondary_category']) + (
                    len(x['secondary_category']) + len(x['type']) -
                    len(x['category'])
            ), axis=1) > 0
        secondary_category = self.category_and_subcategories.loc[
            f_flipped, 'secondary_category']
        type_category = self.category_and_subcategories.loc[f_flipped, 'type']

        self.category_and_subcategories.loc[
            f_flipped, 'secondary_category'] = type_category
        self.category_and_subcategories.loc[
            f_flipped, 'type'] = secondary_category

        return self.category_and_subcategories

    def __create_super_categories(self) -> pd.DataFrame:
        """
        Add a column with handmade super-categories
        :return:
        """

        # Create super-categories
        self.category_and_subcategories['supercategories'] = \
            self.category_and_subcategories['main_category'].map(
                supercategories)

        return self.category_and_subcategories

    def __handmade_category_corrections(self) -> pd.DataFrame:
        """
        Move part of the 'main_category' name to the 'secondary_category' name
        :return:
        """

        for original_category in category_corrections.keys():
            f_original_category = \
                self.category_and_subcategories.main_category ==\
                original_category
            self.category_and_subcategories.loc[
                f_original_category, 'main_category'] = \
                category_corrections[original_category]
            self.category_and_subcategories.loc[
                f_original_category, 'secondary_category'] = \
                original_category[
                len(category_corrections[original_category]) + 1:] + '_' + \
                self.category_and_subcategories.loc[
                    f_original_category, 'secondary_category']

        return self.category_and_subcategories

    def __simplify_dataset(self,
                           only_full_range: bool = False) -> pd.DataFrame:
        """
        Reduces de dataset to keep only useful data for visualization by:
        1. Dropping unnecessary columns
        2. Keeping only one value per month
        3. Keeping only IDs with data each month (optional)

        :param only_full_range: Keeps only IDs with data each month
        :return:
        """
        # Drop unnecessary columns for visualization
        self.dataset = self.dataset.drop(
            columns=['url', 'category', 'description'])

        # Group by month
        self.dataset = self.dataset.groupby(['insert_date', 'product_id']
                                            ).last()
        self.dataset = self.dataset.reset_index()

        # Drop timeseries without full daterange
        if only_full_range:
            id_counts = self.dataset.product_id.value_counts()
            f_max_counts = id_counts == id_counts.max()
            id_full_timeseries = id_counts[f_max_counts].index.values
            f_full_timeseries = \
                self.dataset.product_id.isin(id_full_timeseries)
            self.dataset = self.dataset[f_full_timeseries]

        return self.dataset

    def __add_price_diff(self) -> pd.DataFrame:
        """
        Add 1 year price diff to the max date products entry when possible
        :return:
        """
        year = self.dataset.insert_date.max().year
        month = self.dataset.insert_date.max().month
        day = self.dataset.insert_date.max().day

        f_last = self.dataset.insert_date == f'{year}-{month}-{day}'
        f_first = self.dataset.insert_date == f'{year - 1}-{month}-{day}'

        price_diff = self.dataset.loc[f_first | f_last].pivot_table(
            values='price',
            index='product_id',
            columns='insert_date')
        price_diff = price_diff.iloc[:, 1] - price_diff.iloc[:, 0]

        df_price_diff = pd.DataFrame({'price_diff': price_diff,
                                      'insert_date':
                                          pd.to_datetime(f'{year}-{month}')})
        df_price_diff.insert_date = df_price_diff.insert_date.dt.to_period('M')
        df_price_diff = df_price_diff.reset_index()
        self.dataset = pd.merge(self.dataset, df_price_diff, how='left')

        return self.dataset

    def __remove_wrong_extreme_prices(self) -> pd.DataFrame:
        """
        Correct specific Carrefour and Mercadona data mistakes.
        * Carrefour got 'reference_price' 1000 times bigger than it should be
        for products with 'ud' as reference_unit
        * Mercadona got 'price' 99 times bigger than it should be
        for products with 'kg' as reference_unit
        :return:
        """

        # Wrong Carrefour reference prices
        f_carrefour = self.dataset.supermarket == 'carrefour-es'
        f_ud = self.dataset.reference_unit == 'ud'
        f_ratio = \
            self.dataset.reference_price / self.dataset.price == 1000
        unique_wrong_ids = \
            self.dataset[f_carrefour & f_ud & f_ratio].product_id.unique()

        self.dataset.loc[
            self.dataset.product_id.isin(unique_wrong_ids),
            'reference_price'] = self.dataset.loc[
            self.dataset.product_id.isin(unique_wrong_ids),
            'price']

        # Wrong Mercadona prices
        f_mercadona = self.dataset.supermarket == 'mercadona-es'
        f_kg = self.dataset.reference_unit == 'kg'
        f_ratio = \
            self.dataset.price / self.dataset.reference_price == 99
        unique_wrong_ids = \
            self.dataset[f_mercadona & f_kg & f_ratio].product_id.unique()

        self.dataset.loc[
            self.dataset.product_id.isin(unique_wrong_ids),
            'price'] = self.dataset.loc[
            self.dataset.product_id.isin(unique_wrong_ids), 'reference_price']

        return self.dataset


if __name__ == '__main__':
    df = pd.read_csv('datamarket_productos_de_supermercados.csv')
    sc = SupermarketCategories(df, load_split_categories=False)
    sc.dataset.to_csv('datamarket.csv', index=False)
