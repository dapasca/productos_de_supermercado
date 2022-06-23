from typing import List

import pandas as pd
import numpy as np


class SupermarketCategories:
    def __init__(self,
                 dataset: pd.DataFrame,
                 split_categories_path: str = 'split_categories.csv'):
        """
        Operate with "Productos de supermercados" dataset
        from https://datamarket.es

        :param dataset: the dataset from https://datamarket.es
        :param split_categories_path: The path to the categories.
        If None categories will be estimated from self.dataset
        """

        self.dataset = dataset
        self.dataset['insert_date'] = \
            pd.to_datetime(self.dataset['insert_date'])

        if split_categories_path is not None:
            self.category_and_subcategories = \
                pd.read_csv(split_categories_path)
        else:
            self.category_and_subcategories = self.iterative_split()

        self.dataset = pd.merge(
            self.dataset, self.category_and_subcategories, on='category')

    @staticmethod
    def __split_with_junctions(category: str) -> List[str]:
        """
        Split a string by the '_' but
        ignoring the '_' surrounding junctions

        :param category: The string to be split
        :return: Split string
        """
        junction_list = ['y', 'e', 'de', 'del', 'para',
                         'el', 'la', 'en', 'con', 'sin']

        for junction in junction_list:
            category = \
                ('_'+category).replace(f'_{junction}_', f' {junction} ')[1:]
            category = \
                ('_'+category).replace(f' {junction}_', f' {junction} ')[1:]
            category = \
                ('_'+category).replace(f'_{junction} ', f' {junction} ')[1:]
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
        category_list = \
            list(map(lambda s: s.replace('__', '_'), category_list))

        left_list = []
        right_list = []

        for i, cl in enumerate(category_list):

            left_category = None
            right_category = None

            category_after = self.__split_with_junctions(category_list[i - 1])
            category = self.__split_with_junctions(category_list[i])
            category_before = \
                self.__split_with_junctions(
                    category_list[i + 1]) if len(category_list) > i+1 else []

            # Categories with repeated text split by repeated pattern
            if len(category) > 1:
                for w_i in range(1, (len(category) // 2)+1):
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
                        if w_i+1 < len(category):
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
        self.flip_wrong_columns()

        if save:
            self.category_and_subcategories.to_csv('split_categories.csv',
                                                   index=False)
        return self.category_and_subcategories

    def flip_wrong_columns(self):
        """
        If columns are flipped, correct it
        :return:
        """
        f_fliped = self.category_and_subcategories.apply(
            lambda x: x['category'].find(x['secondary_category']) +
                      len(x['secondary_category']) + len(x['type']) -
                      len(x['category']),
            axis=1) > 0
        secondary_category = self.category_and_subcategories.loc[
            f_fliped, 'secondary_category']
        type_category = self.category_and_subcategories.loc[f_fliped, 'type']

        self.category_and_subcategories.loc[
            f_fliped, 'secondary_category'] = type_category
        self.category_and_subcategories.loc[
            f_fliped, 'type'] = secondary_category


if __name__ == '__main__':
    df = pd.read_csv('datamarket_productos_de_supermercados.csv')
    sc = SupermarketCategories(df, split_categories_path=None)
