import numpy as np
import pandas as pd
import os
import sys
import argparse
from dataset_preprocessing_forecast import ETT_processing, traffic_processing, weather_processing


abs_path = os.path.dirname(os.path.realpath(__file__))




def fcst_preprocessing(dataset, ot_granu):
    assert dataset in ["ETTm1", "ETTm2", "traffic", "weather"]

    if dataset in ["ETTm1", "ETTm2"] :
        df_ori = pd.read_csv(f"{abs_path}/ETT/" + dataset + ".csv")
    else :
        df_ori = pd.read_csv(f"{abs_path}/" + dataset + "/" + dataset + "/" + ".csv")
    
    if dataset in ["ETTm1", "ETTm2"] :
        resample_df = ETT_processing(dataset, ot_granu, df_ori)
    elif dataset == "traffic" :
        resample_df = traffic_processing(ot_granu, df_ori)
    elif dataset == "weather" :
        resample_df = weather_processing(ot_granu, df_ori)
    else : raise
    processed_to_middata(resample_df, dataset, ot_granu)


def processed_to_middata(resample_df, data, OT_granu):
    df = resample_df.copy()
    if data == "ETTm1" or data == "ETTm2":
        assert OT_granu in ["quarterly", "hourly", "daily"]
        if OT_granu == "quarterly" :
            resample_df = df
            resample_df.to_csv(f"{abs_path}/PUBLIC_DATASETS/{data}_processed_OT={OT_granu}.csv", index=False)
        elif OT_granu == "hourly" :
            resample_df = pd.DataFrame()
            col_idx = 0
            for each_col in df.columns.tolist():
                if each_col in ["HUFL", "HULL"] :
                    resample_col = df[each_col].values.reshape(-1, 4)
                    for j in range(4):
                        resample_df[str(col_idx)] = resample_col[ : , j]
                        col_idx += 1
                elif each_col in ["MUFL", "MULL", "LUFL", "LULL"] :
                    col_val = df[each_col].values.tolist()
                    resample_col = [col_val[j] for j in range(0, len(col_val), 4)]
                    resample_df[str(col_idx)] = resample_col
                    col_idx += 1
                elif each_col == "OT" :
                    col_val = df[each_col].values.tolist()
                    resample_col = [col_val[j] for j in range(0, len(col_val), 4)]
                    resample_df["OT"] = resample_col
                elif each_col == "date" :
                    col_val = df[each_col].values.tolist()
                    resample_col = [col_val[j] for j in range(0, len(col_val), 4)]
                    resample_df["date"] = resample_col
                else : raise
            resample_df.to_csv(f"{abs_path}/PUBLIC_DATASETS/{data}_processed_OT={OT_granu}.csv", index=False)    
        elif OT_granu == "daily" :
            resample_df = pd.DataFrame()
            col_idx = 0
            df = df.iloc[ : int(df.shape[0] / 96) * 96, : ]
            for each_col in df.columns.tolist():
                if each_col in ["HUFL", "HULL"] :
                    resample_col = df[each_col].values.reshape(-1, 96)
                    for j in range(96):
                        resample_df[str(col_idx)] = resample_col[ : , j]
                        col_idx += 1
                elif each_col in ["MUFL", "MULL"] :
                    col_val = df[each_col].values.tolist()
                    resample_col = [col_val[j] for j in range(0, len(col_val), 4)]
                    resample_col = np.array(resample_col).reshape(-1, 24)
                    for j in range(24):
                        resample_df[str(col_idx)] = resample_col[ : , j]
                        col_idx += 1
                elif each_col in ["LUFL", "LULL"] :
                    col_val = df[each_col].values.tolist()
                    resample_col = [col_val[j] for j in range(0, len(col_val), 96)]
                    resample_df[str(col_idx)] = resample_col
                    col_idx += 1
                elif each_col == "OT" :
                    col_val = df[each_col].values.tolist()
                    resample_col = [col_val[j] for j in range(0, len(col_val), 96)]
                    resample_df["OT"] = resample_col
                elif each_col == "date" :
                    col_val = df[each_col].values.tolist()
                    resample_col = [col_val[j] for j in range(0, len(col_val), 96)]
                    resample_df["date"] = resample_col
                else : raise
            resample_df.to_csv(f"{abs_path}/PUBLIC_DATASETS/{data}_processed_OT={OT_granu}.csv", index=False)

    elif data == "weather" :
        assert OT_granu in ["10-minute", "hourly", "daily", "weekly"]
        all_cols = df.columns.tolist()
        hourly_cols = ["VPmax (mbar)", "VPact (mbar)", "VPdef (mbar)"]
        daily_cols = ["sh (g/kg)", "H2OC (mmol/mol)", "rho (g/m**3)"]
        minute_cols = list(set(all_cols) - set(hourly_cols) - set(daily_cols) - {"date"} - {"OT"})
        if OT_granu == "10-minute" :
            resample_df = df
            resample_df.to_csv(f"{abs_path}/PUBLIC_DATASETS/{data}_processed_OT={OT_granu}.csv", index=False)
        elif OT_granu == "hourly" :
            resample_df = pd.DataFrame()
            col_idx = 0
            df = df.iloc[ : int(df.shape[0] / 6) * 6, : ]
            for each_col in df.columns.tolist():
                if each_col in minute_cols :
                    resample_col = df[each_col].values.reshape(-1, 6)
                    for j in range(4):
                        resample_df[str(col_idx)] = resample_col[ : , j]
                        col_idx += 1
                elif each_col in hourly_cols or each_col in daily_cols :
                    col_val = df[each_col].values.tolist()
                    resample_col = [col_val[j] for j in range(0, len(col_val), 6)]
                    resample_df[str(col_idx)] = resample_col
                    col_idx += 1
                elif each_col == "OT" :
                    col_val = df[each_col].values.tolist()
                    resample_col = [col_val[j] for j in range(0, len(col_val), 6)]
                    resample_df["OT"] = resample_col
                elif each_col == "date" :
                    col_val = df[each_col].values.tolist()
                    resample_col = [col_val[j] for j in range(0, len(col_val), 6)]
                    resample_df["date"] = resample_col
                else : raise
            resample_df.to_csv(f"{abs_path}/PUBLIC_DATASETS/{data}_processed_OT={OT_granu}.csv", index=False)
        elif OT_granu == "daily" :
            resample_df = pd.DataFrame()
            col_idx = 0
            df = df.iloc[ : int(df.shape[0] / 144) * 144, : ]
            for each_col in df.columns.tolist():
                if each_col in minute_cols :
                    resample_col = df[each_col].values.reshape(-1, 144)
                    for j in range(144):
                        resample_df[str(col_idx)] = resample_col[ : , j]
                        col_idx += 1
                elif each_col in hourly_cols :
                    col_val = df[each_col].values.tolist()
                    resample_col = [col_val[j] for j in range(0, len(col_val), 6)]
                    resample_col = np.array(resample_col).reshape(-1, 24)
                    for j in range(24):
                        resample_df[str(col_idx)] = resample_col[ : , j]
                        col_idx += 1
                elif each_col in daily_cols :
                    col_val = df[each_col].values.tolist()
                    resample_col = [col_val[j] for j in range(0, len(col_val), 144)]
                    resample_df[str(col_idx)] = resample_col
                    col_idx += 1
                elif each_col == "OT" :
                    col_val = df[each_col].values.tolist()
                    resample_col = [col_val[j] for j in range(0, len(col_val), 144)]
                    resample_df["OT"] = resample_col
                elif each_col == "date" :
                    col_val = df[each_col].values.tolist()
                    resample_col = [col_val[j] for j in range(0, len(col_val), 144)]
                    resample_df["date"] = resample_col
                else : raise
            resample_df.to_csv(f"{abs_path}/PUBLIC_DATASETS/{data}_processed_OT={OT_granu}.csv", index=False)
        elif OT_granu == "weekly" :
            resample_df = pd.DataFrame()
            col_idx = 0
            df = df.iloc[ : int(df.shape[0] / 1008) * 1008, : ]
            for each_col in df.columns.tolist():
                if each_col in minute_cols :
                    resample_col = df[each_col].values.reshape(-1, 1008)
                    for j in range(1008):
                        resample_df[str(col_idx)] = resample_col[ : , j]
                        col_idx += 1
                elif each_col in hourly_cols :
                    col_val = df[each_col].values.tolist()
                    resample_col = [col_val[j] for j in range(0, len(col_val), 6)]
                    resample_col = np.array(resample_col).reshape(-1, 168)
                    for j in range(168):
                        resample_df[str(col_idx)] = resample_col[ : , j]
                        col_idx += 1
                elif each_col in daily_cols :
                    col_val = df[each_col].values.tolist()
                    resample_col = [col_val[j] for j in range(0, len(col_val), 1008)]
                    resample_df[str(col_idx)] = resample_col
                    col_idx += 1
                elif each_col == "OT" :
                    col_val = df[each_col].values.tolist()
                    resample_col = [col_val[j] for j in range(0, len(col_val), 1008)]
                    resample_df["OT"] = resample_col
                elif each_col == "date" :
                    col_val = df[each_col].values.tolist()
                    resample_col = [col_val[j] for j in range(0, len(col_val), 1008)]
                    resample_df["date"] = resample_col
                else : raise
            # print(resample_df['OT'])
            resample_df.to_csv(f"{abs_path}/PUBLIC_DATASETS/{data}_processed_OT={OT_granu}.csv", index=False)
    
    elif data == "traffic":
        assert OT_granu in ["hourly", "daily"]
        if OT_granu == "hourly" :
            resample_df = df
            resample_df.to_csv(f"{abs_path}/PUBLIC_DATASETS/{data}_processed_OT={OT_granu}.csv", index=False)
        elif OT_granu == "daily" :
            resample_df = pd.DataFrame()
            col_idx = 0
            df = df.iloc[ : int(df.shape[0] / 24) * 24, : ]
            for each_col in df.columns.tolist():
                if each_col not in ["OT", "date"] :
                    resample_col = df[each_col].values.tolist()
                    resample_col = np.array(resample_col).reshape(-1, 24)
                    for j in range(24):
                        resample_df[str(col_idx)] = resample_col[ : , j]
                        col_idx += 1
                elif each_col == "OT" :
                    col_val = df[each_col].values.tolist()
                    resample_col = [col_val[j] for j in range(0, len(col_val), 24)]
                    resample_df["OT"] = resample_col
                elif each_col == "date" :
                    col_val = df[each_col].values.tolist()
                    resample_col = [col_val[j] for j in range(0, len(col_val), 24)]
                    resample_df["date"] = resample_col
                else : raise
            # print(resample_df['OT'])
            resample_df.to_csv(f"{abs_path}/PUBLIC_DATASETS/{data}_processed_OT={OT_granu}.csv", index=False)




if __name__ == "__main__" :

    os.makedirs(f"{abs_path}/PUBLIC_DATASETS/", exist_ok=True)

    parser = argparse.ArgumentParser(description='MF-CLR')
    parser.add_argument('--dataset', type=str, required=True, help='The dataset name, This can be set to ETTm1, ETTm2, traffic, weather')
    parser.add_argument('--ot_granu', type=str, required=True, default='quarterly', help='frequency of the forecast target')
    
    args = parser.parse_args()
    fcst_preprocessing(args.dataset, args.ot_granu)