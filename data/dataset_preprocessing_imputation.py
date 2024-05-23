import os
import pandas as pd
import numpy as np


abs_path = os.path.dirname(os.path.realpath(__file__))




def ETT_processing(file_name:str, OT_granu:str, mask_ratio, _style= "all"):

    assert _style in ["all", "data", "index"]
    ett_df = pd.read_csv(f"{abs_path}" + "/ETT/" + file_name + ".csv")
    mask_idx = np.arange(ett_df.shape[0])

    if _style in ["all", "data"] :
        assert OT_granu in ["quarterly", "hourly", "daily"]
        for each_col in ett_df.columns.tolist() :
            if each_col in ["MUFL", "MULL"] :
                each_col_p = []
                each_col_v = ett_df[each_col].values
                for i in range(each_col_v.shape[0]):
                    if i % 4 == 0 :
                        each_col_p.append(each_col_v[i])
                    else :
                        each_col_p.append(each_col_p[i - 1])
                ett_df[each_col] = each_col_p
            elif each_col in ["LUFL", "LULL"] :
                each_col_p = []
                each_col_v = ett_df[each_col].values
                for i in range(each_col_v.shape[0]):
                    if i % 96 == 0 :
                        each_col_p.append(each_col_v[i])
                    else :
                        each_col_p.append(each_col_p[i - 1])
                ett_df[each_col] = each_col_p
            elif each_col == "OT" :
                if OT_granu == "quarterly" : 
                    pass
                elif OT_granu == "hourly" : 
                    each_col_p = []
                    each_col_v = ett_df[each_col].values
                    for i in range(each_col_v.shape[0]):
                        if i % 4 == 0 :
                            each_col_p.append(each_col_v[i])
                        else :
                            each_col_p.append(each_col_p[i - 1])
                    ett_df[each_col] = each_col_p
                elif OT_granu == "daily" : 
                    each_col_p = []
                    each_col_v = ett_df[each_col].values
                    for i in range(each_col_v.shape[0]):
                        if i % 96 == 0 :
                            each_col_p.append(each_col_v[i])
                        else :
                            each_col_p.append(each_col_p[i - 1])
                    ett_df[each_col] = each_col_p
    
    if _style in ["all", "index"] :
        mask_num = int(ett_df.shape[0] * mask_ratio)
        np.random.shuffle(mask_idx)
        mask_idx = mask_idx[ : mask_num]
        mask_idx = np.sort(mask_idx)
    
    return ett_df, np.int0(mask_idx)




# every 10 minutes to every 10 minutes, hourly, daily, and weekly
def weather_processing(OT_granu:str, mask_ratio, _style= "all"):

    assert _style in ["all", "data", "index"]
    weather_df = pd.read_csv(f"{abs_path}" + "/weather/weather.csv")
    mask_idx = np.arange(weather_df.shape[0])


    if _style in ["all", "data"] :
        assert OT_granu in ["10-minute", "hourly", "daily", "weekly"]
        col_list = weather_df.columns.tolist()
        for col_idx in range(weather_df.shape[1]):
            weather_df.rename(columns= {col_list[col_idx] : col_list[col_idx].split(" (")[0]}, inplace= True)

        for each_col in weather_df.columns.tolist():
            if each_col in ["date", "p", "T", "Tpot", "Tdew"] :
                pass
            elif each_col in ["rh", "VPmax", "VPact", "VPdef", "sh", "H2OC", "rho"] :
                each_col_p = []
                each_col_v = weather_df[each_col].values
                for i in range(each_col_v.shape[0]):
                    if i % 6 == 0 :
                        each_col_p.append(each_col_v[i])
                    else :
                        each_col_p.append(each_col_p[i - 1])
                weather_df[each_col] = each_col_p
            elif each_col in ["wv", "max. wv", "wd"] :
                each_col_p = []
                each_col_v = weather_df[each_col].values
                for i in range(each_col_v.shape[0]):
                    if i % 144 == 0 :
                        each_col_p.append(each_col_v[i])
                    else :
                        each_col_p.append(each_col_p[i - 1])
                weather_df[each_col] = each_col_p
            elif each_col in ['rain', 'raining', 'SWDR', 'PAR', 'max. PAR', 'Tlog'] :
                each_col_p = []
                each_col_v = weather_df[each_col].values
                for i in range(each_col_v.shape[0]):
                    if i % 1008 == 0 :
                        each_col_p.append(np.mean(each_col_v[i : i + 1008]))
                    else :
                        each_col_p.append(each_col_p[i - 1])
                weather_df[each_col] = each_col_p
            
            elif each_col == "OT" :
                if OT_granu == "10-minute" : 
                    pass
                elif OT_granu == "hourly" : 
                    each_col_p = []
                    each_col_v = weather_df[each_col].values
                    for i in range(each_col_v.shape[0]):
                        if i % 6 == 0 :
                            each_col_p.append(each_col_v[i])
                        else :
                            each_col_p.append(each_col_p[i - 1])
                    weather_df[each_col] = each_col_p
                elif OT_granu == "daily" : 
                    each_col_p = []
                    each_col_v = weather_df[each_col].values
                    for i in range(each_col_v.shape[0]):
                        if i % 144 == 0 :
                            each_col_p.append(each_col_v[i])
                        else :
                            each_col_p.append(each_col_p[i - 1])
                    weather_df[each_col] = each_col_p
                elif OT_granu == "weekly" :
                    each_col_p = []
                    each_col_v = weather_df[each_col].values
                    for i in range(each_col_v.shape[0]):
                        if i % 1008 == 0 :
                            each_col_p.append(np.mean(each_col_v[i : i + 1008]))
                        else :
                            each_col_p.append(each_col_p[i - 1])
                    weather_df[each_col] = each_col_p
            else : 
                print(each_col)
                raise
    

    if _style in ["all", "index"] :
        mask_num = int(weather_df.shape[0] * mask_ratio)
        np.random.shuffle(mask_idx)
        mask_idx = mask_idx[ : mask_num]
        mask_idx = np.sort(mask_idx)

    return weather_df, np.int0(mask_idx)




# every hour to hourly and daily
def traffic_processing(OT_granu:str, mask_ratio, _style= "all"):

    assert _style in ["all", "data", "index"]
    traffic_df = pd.read_csv(f"{abs_path}" + "/traffic/traffic.csv")
    mask_idx = np.arange(traffic_df.shape[0])

    if _style in ["all", "data"] :
        assert OT_granu in ["hourly", "daily"]
        col_list = traffic_df.columns.tolist()
        for col_idx in range(401, traffic_df.shape[1] - 1):
            each_col = col_list[col_idx]
            each_col_p = []
            each_col_v = traffic_df[each_col].values
            for i in range(each_col_v.shape[0]):
                if i % 24 == 0 :
                    each_col_p.append(np.mean(each_col_v[i : i + 24]))
                else :
                    each_col_p.append(each_col_p[i - 1])
            traffic_df[each_col] = each_col_p
        if OT_granu == "hourly" :
            pass
        elif OT_granu == "daily" :
            each_col_p = []
            each_col_v = traffic_df["OT"].values
            for i in range(each_col_v.shape[0]):
                if i % 24 == 0 :
                    each_col_p.append(np.mean(each_col_v[i : i + 24]))
                else :
                    each_col_p.append(each_col_p[i - 1])
            traffic_df["OT"] = each_col_p

    if _style in ["all", "index"] :
        mask_num = int(traffic_df.shape[0] * mask_ratio)
        np.random.shuffle(mask_idx)
        mask_idx = mask_idx[ : mask_num]
        mask_idx = np.sort(mask_idx)

    return traffic_df, np.int0(mask_idx)




if __name__ == "__main__" :

    _, mask_idx = weather_processing("hourly", 0.5, "index")
    print(mask_idx[:20])
    for r in mask_idx : print(r)
