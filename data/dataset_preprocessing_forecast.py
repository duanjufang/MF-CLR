import pandas as pd
import numpy as np




def ETT_processing(file_name:str, OT_granu:str, input_df= None):
    
    # hourly to hourly daily weekly
    if file_name == "ETTh1" or file_name == "ETTh2" :
        assert OT_granu in ["hourly", "daily", "weekly"]
        if input_df is None :
            ett_df = pd.read_csv("PUBLIC_DATASETS/" + file_name + ".csv")
        else :
            ett_df = input_df.copy()
        for each_col in ett_df.columns.tolist() :
            if each_col in ["MUFL", "MULL"] :
                each_col_p = []
                each_col_v = ett_df[each_col].values
                for i in range(each_col_v.shape[0]):
                    if i % 24 == 0 :
                        each_col_p.append(each_col_v[i])
                    else :
                        each_col_p.append(each_col_p[i - 1])
                ett_df[each_col] = each_col_p
            elif each_col in ["LUFL", "LULL"] :
                each_col_p = []
                each_col_v = ett_df[each_col].values
                for i in range(each_col_v.shape[0]):
                    if i % 168 == 0 :
                        each_col_p.append(each_col_v[i])
                    else :
                        each_col_p.append(each_col_p[i - 1])
                ett_df[each_col] = each_col_p
            elif each_col == "OT" :
                if OT_granu == "hourly" : 
                    pass
                elif OT_granu == "daily" : 
                    each_col_p = []
                    each_col_v = ett_df[each_col].values
                    for i in range(each_col_v.shape[0]):
                        if i % 24 == 0 :
                            each_col_p.append(each_col_v[i])
                        else :
                            each_col_p.append(each_col_p[i - 1])
                    ett_df[each_col] = each_col_p
                elif OT_granu == "weekly" : 
                    each_col_p = []
                    each_col_v = ett_df[each_col].values
                    for i in range(each_col_v.shape[0]):
                        if i % 168 == 0 :
                            each_col_p.append(each_col_v[i])
                        else :
                            each_col_p.append(each_col_p[i - 1])
                    ett_df[each_col] = each_col_p
    
    # quarterly to quarterly hourly daily
    elif file_name == "ETTm1" or file_name == "ETTm2" :
        assert OT_granu in ["quarterly", "hourly", "daily"]
        if input_df is None :
            ett_df = pd.read_csv("PUBLIC_DATASETS/" + file_name + ".csv")
        else :
            ett_df = input_df.copy()
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
    
    return ett_df




# every 10 minutes to every 10 minutes, hourly, daily, and weekly
def weather_processing(OT_granu:str, input_df= None):
    assert OT_granu in ["10-minute", "hourly", "daily", "weekly"]
    if input_df is None :
        weather_df = pd.read_csv("PUBLIC_DATASETS/weather.csv")
    else : 
        weather_df = input_df.copy()
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
    return weather_df




# every hour to hourly and daily
def traffic_processing(OT_granu:str, input_df= None):
    assert OT_granu in ["hourly", "daily"]
    if input_df is None :
        traffic_df = pd.read_csv("PUBLIC_DATASETS/traffic.csv")
    else : 
        traffic_df = input_df.copy()
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
    return traffic_df




if __name__ == "__main__" :

    # for file_name in ["ETTm1", "ETTm2", "ETTh1", "ETTh2"] : 
    #     if file_name in ["ETTh1", "ETTh2"] :
    #         for OT_granu in ["hourly", "daily", "weekly"] : 
    #             processed_df = ETT_processing(file_name, OT_granu)
    #             processed_df.to_csv("PUBLIC_DATASETS/" + file_name + "_proceesed_OT=" + OT_granu + ".csv", index= False)
    #     elif file_name in ["ETTm1", "ETTm2"] :
    #         for OT_granu in ["quarterly", "hourly", "daily"] : 
    #             processed_df = ETT_processing(file_name, OT_granu)
    #             processed_df.to_csv("PUBLIC_DATASETS/" + file_name + "_proceesed_OT=" + OT_granu + ".csv", index= False)


    for OT_granu in ["10-minute", "hourly", "daily", "weekly"] : 
        processed_df = weather_processing(OT_granu)
        processed_df.to_csv("PUBLIC_DATASETS/weather_proceesed_OT=" + OT_granu + ".csv", index= False)


    # for OT_granu in ["hourly", "daily"] : 
    #     processed_df = traffic_processing(OT_granu)
    #     processed_df.to_csv("PUBLIC_DATASETS/traffic_proceesed_OT=" + OT_granu + ".csv", index= False)