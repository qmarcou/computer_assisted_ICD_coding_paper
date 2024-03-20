"""Utilities to navigate among ontologies/OMOP concepts."""
from __future__ import annotations

import copy
import numpy as np
import pandas as pd
import dtypecheck as dt


def apply_relationship(df, concept_colname, relationship,
                       concept_relationship, output_col=None,
                       inner=False) -> pd.DataFrame:
    """
    Apply a relationship to a column.

    The new output column has the name of the relationship unless specified
    otherwise.
    """
    if inner:
        how = "inner"
    else:
        how = "left"
    concept_relationship = concept_relationship.query('relationship_id=='
                                                      + '"' + relationship
                                                      + '"')
    concept_relationship.rename(columns={'concept_id_2': '_tp_concept_id_2',
                                         'concept_id_1': '_tp_concept_id_1'},
                                inplace=True)
    merged_df = pd.merge(df,
                         concept_relationship[[
                             '_tp_concept_id_1', '_tp_concept_id_2']],
                         left_on=concept_colname, right_on='_tp_concept_id_1',
                         how=how)
    merged_df.drop(columns='_tp_concept_id_1', inplace=True)
    if output_col is None:
        merged_df.rename(columns={'_tp_concept_id_2': relationship},
                         inplace=True)
    else:
        merged_df.rename(columns={'_tp_concept_id_2': output_col},
                         inplace=True)

    return merged_df


def map_to_rxnorm_ingredient(drugs_df: pd.DataFrame,
                             concept_relationship: pd.DataFrame,
                             concepts: pd.DataFrame,
                             strict: bool = False,
                             drug_concept_id_col: str = "drug_concept_id") -> pd.DataFrame:
    """
    Map RxNorm codes to their RxNorm ingredient.

    Maps RxNorm concepts contained in the DF of interest to ingredient.

    Parameters:
    drugs_df : pd.DataFrame
        a DataFrame containing a column with RxNorm concepts
    concept_relationship : pd.DataFrame
        a DataFrame containing RxNorm concept relationship(or more).
        If you provide filtered concept relationship make sure you provide
        all relevant relationships (including RxNorm extensions).
    concepts : pd.DataFrame
        a DataFrame containing RxNorm concepts(or more).
    strict : Boolean
        If True will only return Ingredient concepts or NaN, if False the
        function will return concept class closest to Ingredient in case a
        concept could not be mapped to Ingredient.
    drug_concept_id_col: string
        Name of the column containing the RxNorm concepts to be mapped.

    Returns:
        Pandas.Dataframe: a Dataframe with an extra column containing mapped RxNorm
        concepts

    """
    # Sequence of relationships to apply in order to obtain RxNorm ingredient
    # based on https://www.nlm.nih.gov/research/umls/rxnorm/RxNorm_Drug_Relationships.png
    relationships = ["Tradename of", "Contains", "Quantified form of",
                     "Consists of", "RxNorm has ing", "Form of"]

    # pre filter relationships and concepts for efficiency
    concept_relationship = concept_relationship.query(
        'relationship_id in ' + str(relationships))
    concepts = concepts.query("concept_class_id == 'Ingredient'")

    drugs_df.loc[:, 'prev_tmp_col'] = drugs_df.loc[:, drug_concept_id_col]

    for i, relationship in enumerate(relationships):
        drugs_df = apply_relationship(drugs_df, "prev_tmp_col", relationship,
                                      concept_relationship,
                                      output_col="work_tmp_col")
        if strict & i == len(relationships) - 1:
            pass
        else:
            # In case the source concept might not be precise enough for then
            # considered relationship to operate we need to refill
            # Nan created values
            drugs_df['prev_tmp_col'] = drugs_df['work_tmp_col'].fillna(
                drugs_df['prev_tmp_col'], inplace=False)
            drugs_df.drop(columns='work_tmp_col', inplace=True)
        # print(drugs_df.work_tmp_col.isna().sum())

    # Reassign values to concept_id that were already ingredients
    mask = drugs_df['drug_concept_id'].isin(concepts.concept_id)
    drugs_df.loc[mask, 'prev_tmp_col'] = drugs_df.loc[mask, 'drug_concept_id']
    # rename working column
    drugs_df.rename(
        columns={'prev_tmp_col': "rxnorm_ingredient"}, inplace=True)
    return drugs_df


def unroll_relationship(df: pd.DataFrame,
                        concept_colname: str, relationship: str,
                        concept_relationship: pd.DataFrame,
                        max_depth: int = -1,
                        append_to_df: bool = True) -> pd.DataFrame:
    """
    Apply a relationship to a column and append the resulting rows to df.

    New rows are identical to the original ones except for the concept_id
    columns.

    Parameters
    ----------
    df: input dataframe to expand
    concept_colname: name of the column containing the concept to expand
    relationship: relationship_id
    concept_relationship: Dataframe containing the concept relationships
    max_depth: maximum depth to unroll, positive integer, a negative value
    means 'infinite' unroll
    append_to_df: If True returns the original df with the unrolled value
        appended, if False only return the unrolled values.

    Returns
    -------
    pd.Dataframe
    """
    concept_relationship = concept_relationship.query('relationship_id=='
                                                      + '"' + relationship
                                                      + '"')
    if append_to_df:
        concat_df = [df]
    else:
        concat_df = []
    ancestor_df = copy.copy(df)
    depth: int = 0
    while (not ancestor_df.empty) and (max_depth <= 0 or depth < max_depth):
        ancestor_df = apply_relationship(ancestor_df,
                                         concept_colname=concept_colname,
                                         relationship=relationship,
                                         concept_relationship=
                                         concept_relationship,
                                         output_col="_tmp_relat_col",
                                         inner=True)
        ancestor_df.drop(columns=concept_colname, inplace=True)
        ancestor_df.rename(columns={'_tmp_relat_col': concept_colname},
                           inplace=True)
        concat_df.append(ancestor_df)
        depth += 1
    return pd.concat(concat_df)


def is_hier_relationship_redundant(concept_relationship_sub: pd.DataFrame,
                                   concept_relationship_full: pd.DataFrame,
                                   relationship: str,
                                   target_col: str) -> pd.DataFrame:
    """
    Check if the relationships in sub can be obtained from other relationships.

    Parameters
    ----------
    concept_relationship_sub: a sub ensemble of relationships to test
    concept_relationship_full: the larger ensemble of relationships
    relationship: relationship_id
    target_col: the relationship column to unroll

    Returns
    -------
    Boolean pd.Series indicating whether the relationship is a
    redundant/indirect one.
    """
    hier_unroll = unroll_relationship(df=concept_relationship_sub,
                                      concept_colname=target_col,
                                      relationship=relationship,
                                      concept_relationship=concept_relationship_full,
                                      max_depth=-1,
                                      append_to_df=False)
    # Concatenate inital rows and their unroll and check if they are duplicated
    concat_df = pd.concat([concept_relationship_sub, hier_unroll])
    return (concat_df.duplicated(subset=['concept_id_1', 'concept_id_2',
                                         'relationship_id'], keep=False)
                .iloc[0:concept_relationship_sub.shape[0]])


def hierarchy_dag_to_tree(concept_relationship: pd.DataFrame,
                          relationship: str,
                          concept_id_col: int) -> pd.DataFrame:
    """
    Takes a hierarchical relationship and prune edges skipping nodes.

    Prune all "redundant" relationships, meaning all relationships entailed by
    other finer grained relationships
    Examples:
    concept_id_1 concept_id_2 relationship_id
    A B Is a
    A C Is a
    B C Is a

    The second row is redundant, and will be removed.

    Parameters
    ----------
    concept_relationship: concept relationship dataframe
    relationship: relationship_id
    concept_id_col: 1 or 2, indicates the grouping column

    Returns
    -------
    concept relationship data frame

    """
    concept_relationship_wc: pd.DataFrame = concept_relationship.query(
        'relationship_id=='
        + '"' + relationship
        + '"')

    if concept_id_col == 1:
        pass
    elif concept_id_col == 2:
        concept_relationship_wc = concept_relationship_wc.rename(columns={'concept_id_1': 'concept_id_2',
                                       'concept_id_2': 'concept_id_1'})
    else:
        raise ValueError('Concept_id_col must be 1 or 2.')

    mask = is_hier_relationship_redundant(concept_relationship_wc,
                                          concept_relationship_wc,
                                          relationship=relationship,
                                          target_col='concept_id_2')
    if concept_id_col == 2:
        concept_relationship_wc = concept_relationship_wc.rename(columns={'concept_id_1': 'concept_id_2',
                                       'concept_id_2': 'concept_id_1'})

    return pd.concat([concept_relationship.query('relationship_id!='
                                                 + '"' + relationship
                                                 + '"'),
                      concept_relationship_wc.loc[~mask, :]])


def unroll_hierarchy_tree_to_dag(concept_relationship: pd.DataFrame,
                                 relationship: str) -> pd.DataFrame:
    """
    Unroll the hierarchy tree from a concept_relationship DF.

    Recreates redundant relationships.
    Parameters
    ----------
    concept_relationship
    relationship

    Returns
    -------
    Returns a DF containing only required relationship with all the links.
    """
    concept_relationship_wc: pd.DataFrame = concept_relationship.query(
        'relationship_id=='
        + '"' + relationship
        + '"')
    return unroll_relationship(
        df=concept_relationship_wc,
        concept_colname="concept_id_2",
        relationship=relationship,
        concept_relationship=concept_relationship,
        append_to_df=True)


def relationships_to_adjacency_df(concept_relationship: pd.DataFrame,
                                  relationship: str,
                                  unique_concepts=None):
    """
    Summarize a context relationship df into an adjacency matrix.

    Parameters
    ----------
    concept_relationship: the concept relationship Dataframe
    relationship: the relationship.s to consider
    unique_concepts: a 1d iterable containing all unique concepts expected in
        the adjacency df. Can also be used to specify the desired ordering of
        concepts in the resulting df axes.

    Returns
    -------
    A pd.Dataframe with sparse columns
    """
    if dt.is_str_or_strlist(relationship):
        if isinstance(relationship, str):
            relationship = [relationship]
    else:
        raise ValueError("relationship must be a string or list of string")
    concept_relationship: pd.DataFrame = concept_relationship.query(
        'relationship_id.isin(' + str(relationship) + ')').loc[:,
                                         ['concept_id_1', 'concept_id_2']]
    if unique_concepts is None:
        # Get all unique concepts from the relationships DF
        unique_concepts = (
            pd.concat(
                [pd.Series(concept_relationship['concept_id_1'].unique()),
                 pd.Series(concept_relationship['concept_id_2'].unique())])
                .unique())
    else:
        # Restrict relationships to required concepts
        unique_concepts = list(unique_concepts)
        concept_relationship = concept_relationship.loc[
                               concept_relationship.concept_id_1.isin(
                                   unique_concepts)
                               & concept_relationship.concept_id_2.isin(
                                   unique_concepts), :]
        if concept_relationship.shape[0] == 0:
            # Catch this otherwise it will return an empty df and will
            # cause an error in groupby.apply with missing "name" attribute
            raise RuntimeError("The provided unique_concept list has no "
                               "intersection with concept relationships.")

    # Group relationship based on destination concept
    # I use destination concept to enable easy use of pandas sparse columns
    grouped = concept_relationship.groupby(by="concept_id_2")
    # Define a function to apply to each group returning sparse
    empty_sp_col = pd.Series(np.zeros((len(unique_concepts)), dtype=np.uint8),
                             index=unique_concepts)
    final_df = pd.DataFrame(0, index=unique_concepts, columns=unique_concepts,
                            dtype=pd.SparseDtype(dtype=np.uint8, fill_value=0))

    def conceptsreldf_to_adjacency_df(df: pd.DataFrame,
                                      target_df,
                                      empty_col_template: pd.Series):
        empty_col_template = empty_col_template.copy()
        empty_col_template[df['concept_id_1']] = 1
        target_df.loc[:, df.name] = pd.Series(empty_col_template,
                                              index=unique_concepts,
                                              dtype=pd.SparseDtype(
                                                  dtype=np.uint8,
                                                  fill_value=0))
        return None

    # Apply the function
    grouped.apply(conceptsreldf_to_adjacency_df, target_df=final_df,
                  empty_col_template=empty_sp_col)
    return final_df


def test_relationship_tree_coherence(concepts: pd.DataFrame,
                                     concept_relationships: pd.DataFrame,
                                     relationship: str,
                                     root_concepts: list,
                                     reciprocal_relationship: str = '',
                                     ):
    concept_relationships = concept_relationships.query(
        'relationship_id in ["' +
        str(relationship) + '","' +
        str(reciprocal_relationship) + '"]')
    assert not (concept_relationships.duplicated(subset=['concept_id_1',
                                                         'concept_id_2',
                                                         'relationship_id'])
                .any())
    # Check that the number of links is coherent
    relationships_df = (concept_relationships
                        .query('relationship_id=="' + str(relationship) + '"'))
    assert (relationships_df
        .shape[0]) == concepts.shape[0] - len(root_concepts)

    if reciprocal_relationship != '':
        reciprocals_df = (concept_relationships
            .query(
            'relationship_id=="' + str(reciprocal_relationship) + '"'))
        assert (reciprocals_df
            .shape[0]) == concepts.shape[0] - len(root_concepts)

    # Check that each node is in the tree
    assert concepts['concept_id'].isin(
        pd.concat([relationships_df['concept_id_1'],
                   relationships_df['concept_id_2']]).unique()).all()
    if reciprocal_relationship != '':
        assert concepts['concept_id'].isin(
            pd.concat([reciprocals_df['concept_id_1'],
                       reciprocals_df['concept_id_2']]).unique()).all()

    # If a reciprocal relationship is provided check that a reciprocal match
    # each relationship
    if reciprocal_relationship != '':
        assert pd.concat([relationships_df.loc[:, ['concept_id_1', 'concept_id_2']],
                          (reciprocals_df.rename(
                              columns={'concept_id_1': 'concept_id_2',
                                       'concept_id_2': 'concept_id_1'}))
                         .loc[:,
                          ['concept_id_1', 'concept_id_2']]]).duplicated(
            subset=['concept_id_1', 'concept_id_2'], keep="first").sum() == \
               relationships_df.shape[0]

    return True
