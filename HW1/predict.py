# import torch
import sys
from copy import deepcopy
from glob import glob
import pandas as pd
from tqdm import tqdm
import torch

from models import LSTMNet, LSTMDataset, predict

def preprocess_data(test_folder_path):
    """
    TODO: David implement this functions please
    gets the test folder path (the path conains all the psv files) and return a pandas DataFrame for prediction
    """
    from_dir = test_folder_path
    to_dir = 'test_data_with_na.pkl'
    fill_na_series = True
    fill_na_not_series = False
    sepsis_dict = dict()
    non_sepsis_dict = dict()

    for file in tqdm(glob(from_dir + '/patient_*.psv')):
        temp_df = pd.read_csv(file, sep='|')
        # get patient ID
        patient_ID = file[ file.find("patient_") : file.find(".psv") ]

        if  temp_df.SepsisLabel.sum() > 0:
            first_sepsis_row = temp_df.shape[0] - temp_df.SepsisLabel.sum()
            sepsis_dict[patient_ID] = dict()
            sepsis_dict[patient_ID]['age'] = temp_df.loc[0, 'Age']
            sepsis_dict[patient_ID]['gender'] = temp_df.loc[0, 'Gender']
            sepsis_dict[patient_ID]['unit1'] = 1 if temp_df.loc[0, 'Unit1'] == 1 else 0
            sepsis_dict[patient_ID]['unit2'] = 1 if temp_df.loc[0, 'Unit2'] == 1 else 0
            sepsis_dict[patient_ID]['unknown unit'] = 0 if temp_df.loc[0, 'Unit1'] == 1 else 0 if temp_df.loc[0, 'Unit2'] == 1 else 1
            sepsis_dict[patient_ID]['HospAdmTime'] = temp_df.loc[0, 'HospAdmTime']

            sepsis_dict[patient_ID]['Final ICULOS'] = temp_df.loc[first_sepsis_row, 'ICULOS']

            sepsis_dict[patient_ID]['Not Null Percentages'] = temp_df.iloc[:first_sepsis_row, :].count() / temp_df.iloc[:first_sepsis_row, :].shape[0]
            sepsis_dict[patient_ID]['Means'] = temp_df.iloc[:first_sepsis_row, :].mean()[:-7]
            sepsis_dict[patient_ID]['Vars'] = temp_df.iloc[:first_sepsis_row, :].var()[:-7]

            if fill_na_series:
                sepsis_dict[patient_ID]['HR_series'] = temp_df.loc[:first_sepsis_row, 'HR'].interpolate(method='linear').fillna(method='bfill')
                sepsis_dict[patient_ID]['Resp_series'] = temp_df.loc[:first_sepsis_row, 'Resp'].interpolate(method='linear').fillna(method='bfill')
                sepsis_dict[patient_ID]['MAP_series'] = temp_df.loc[:first_sepsis_row, 'MAP'].interpolate(method='linear').fillna(method='bfill')
                sepsis_dict[patient_ID]['O2Sat_series'] = temp_df.loc[:first_sepsis_row, 'O2Sat'].interpolate(method='linear').fillna(method='bfill')
                sepsis_dict[patient_ID]['SBP_series'] = temp_df.loc[:first_sepsis_row, 'SBP'].interpolate(method='linear').fillna(method='bfill')
            else:
                sepsis_dict[patient_ID]['HR_series'] = temp_df.loc[:first_sepsis_row, 'HR']
                sepsis_dict[patient_ID]['Resp_series'] = temp_df.loc[:first_sepsis_row, 'Resp']
                sepsis_dict[patient_ID]['MAP_series'] = temp_df.loc[:first_sepsis_row, 'MAP']
                sepsis_dict[patient_ID]['O2Sat_series'] = temp_df.loc[:first_sepsis_row, 'O2Sat']
                sepsis_dict[patient_ID]['SBP_series'] = temp_df.loc[:first_sepsis_row, 'SBP']
        else:
            non_sepsis_dict[patient_ID] = dict()
            non_sepsis_dict[patient_ID]['age'] = temp_df.loc[0, 'Age']
            non_sepsis_dict[patient_ID]['gender'] = temp_df.loc[0, 'Gender']
            non_sepsis_dict[patient_ID]['unit1'] = 1 if temp_df.loc[0, 'Unit1'] == 1 else 0
            non_sepsis_dict[patient_ID]['unit2'] = 1 if temp_df.loc[0, 'Unit2'] == 1 else 0
            non_sepsis_dict[patient_ID]['unknown unit'] = 0 if temp_df.loc[0, 'Unit1'] == 1 else 0 if temp_df.loc[0, 'Unit2'] == 1 else 1
            non_sepsis_dict[patient_ID]['HospAdmTime'] = temp_df.loc[0, 'HospAdmTime']
            
            non_sepsis_dict[patient_ID]['Final ICULOS'] = temp_df.loc[temp_df.shape[0]-1, 'ICULOS']

            non_sepsis_dict[patient_ID]['Not Null Percentages'] = temp_df.count() / temp_df.shape[0]
            non_sepsis_dict[patient_ID]['Means'] = temp_df.mean()[:-7]
            non_sepsis_dict[patient_ID]['Vars'] = temp_df.var()[:-7]

            if fill_na_series:
                non_sepsis_dict[patient_ID]['HR_series'] = temp_df['HR'].interpolate(method='linear').fillna(method='bfill')
                non_sepsis_dict[patient_ID]['Resp_series'] = temp_df['Resp'].interpolate(method='linear').fillna(method='bfill')
                non_sepsis_dict[patient_ID]['MAP_series'] = temp_df['MAP'].interpolate(method='linear').fillna(method='bfill')
                non_sepsis_dict[patient_ID]['O2Sat_series'] = temp_df['O2Sat'].interpolate(method='linear').fillna(method='bfill')
                non_sepsis_dict[patient_ID]['SBP_series'] = temp_df['SBP'].interpolate(method='linear').fillna(method='bfill')
            else:
                non_sepsis_dict[patient_ID]['HR_series'] = temp_df['HR']
                non_sepsis_dict[patient_ID]['Resp_series'] = temp_df['Resp']
                non_sepsis_dict[patient_ID]['MAP_series'] = temp_df['MAP']
                non_sepsis_dict[patient_ID]['O2Sat_series'] = temp_df['O2Sat']
                non_sepsis_dict[patient_ID]['SBP_series'] = temp_df['SBP']

    sepsis_df = pd.DataFrame.from_dict(data=sepsis_dict, orient='index')
    sepsis_df['SepsisLabel'] = 1
    non_sepsis_df = pd.DataFrame.from_dict(data=non_sepsis_dict, orient='index')
    non_sepsis_df['SepsisLabel'] = 0
    print("join dataframe")
    all_df = pd.concat([sepsis_df, non_sepsis_df])

    final_df = deepcopy(all_df[['SepsisLabel','age', 'gender', 'unit1', 'unit2', 'unknown unit', 'HospAdmTime', 'Final ICULOS', 'HR_series', 'Resp_series', 'MAP_series', 'O2Sat_series', 'SBP_series']])
    final_df['Temp_var'] = all_df['Vars'].apply(lambda x: x['Temp'])
    final_df['Temp_mean'] = all_df['Means'].apply(lambda x: x['Temp'])
    final_df['WBC_not_null'] = all_df['Not Null Percentages'].apply(lambda x: x['WBC'])
    final_df['WBC_mean'] = all_df['Means'].apply(lambda x: x['WBC'])
    final_df['Lactate_not_null'] = all_df['Not Null Percentages'].apply(lambda x: x['Lactate'])
    final_df['BaseExcess_not_null'] = all_df['Not Null Percentages'].apply(lambda x: x['BaseExcess'])

    if fill_na_not_series:
        final_df['Temp_var'] = final_df['Temp_var'].fillna(final_df['Temp_var'].mean())
        final_df['Temp_mean'] = final_df['Temp_mean'].fillna(final_df['Temp_mean'].mean())
        final_df['WBC_not_null'] = final_df['WBC_not_null'].fillna(final_df['WBC_not_null'].mean())
        final_df['WBC_mean'] = final_df['WBC_mean'].fillna(final_df['WBC_mean'].mean())
        final_df['Lactate_not_null'] = final_df['Lactate_not_null'].fillna(final_df['Lactate_not_null'].mean())
        final_df['BaseExcess_not_null'] = final_df['BaseExcess_not_null'].fillna(final_df['BaseExcess_not_null'].mean())
    
    # final_df.to_pickle(to_dir)
    return final_df
    
def main():
    test_path = sys.argv[-1]
    training_data_path = 'weights/train_data_with_na.pkl'

    df_train = pd.read_pickle(training_data_path)
    df_test = preprocess_data(test_path)

    df_train = df_train.fillna(df_train.mean())
    df_test = df_test.fillna(df_train.mean())

    test_ds = LSTMDataset(df_test)

    model_concat = LSTMNet(type='concat')
    model_concat.load_state_dict(torch.load('weights/concat_0.69.pt'))

    y_pred_list, y_gt_list, id_list = predict(model_concat, test_ds)

    pred_df = pd.DataFrame(data={'id': id_list, 'prediction': y_pred_list})
    pred_df.to_csv('prediction.csv', index=False)

    


if __name__ == "__main__":
    main()