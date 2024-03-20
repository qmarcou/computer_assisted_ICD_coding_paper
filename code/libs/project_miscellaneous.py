import numpy as np
import hot_encoding as oh


def filter_Sagi_diagnoses(oh_diagnoses_df):
    """Somewhat reproduces ICD9 code selection from Sagi et al 2020."""
    tmp = oh.threshold_oh_abs(oh_diagnoses_df, 100)
    mask = np.logical_not(oh_diagnoses_df.columns.isin(tmp.columns))
    mask = np.logical_or(mask, oh_diagnoses_df.columns.str.startswith('765.'))
    mask = np.logical_or(mask, oh_diagnoses_df.columns.str.startswith('8'))
    mask = np.logical_or(mask, oh_diagnoses_df.columns.str.startswith('9'))
    mask = np.logical_or(mask, oh_diagnoses_df.columns.str.startswith('E'))
    mask = np.logical_or(mask, oh_diagnoses_df.columns.str.startswith('V'))
    return oh_diagnoses_df.columns[mask]

def infrequent_codes_mask(oh_diagnoses_df):
    """bla"""
    tmp = oh.threshold_oh_abs(oh_diagnoses_df, 100)
    mask = np.logical_not(oh_diagnoses_df.columns.isin(tmp.columns))
    return mask

def sagi_filtered_out_roots_mask(oh_diagnoses_df):
    """bla"""
    mask = oh_diagnoses_df.columns.str.startswith('765.')
    mask = np.logical_or(mask, oh_diagnoses_df.columns.str.startswith('8'))
    mask = np.logical_or(mask, oh_diagnoses_df.columns.str.startswith('9'))
    mask = np.logical_or(mask, oh_diagnoses_df.columns.str.startswith('E'))
    mask = np.logical_or(mask, oh_diagnoses_df.columns.str.startswith('V'))
    return mask
