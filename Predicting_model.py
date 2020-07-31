# -*- coding: utf-8 -*-
"""
Created on Fri Jul 31 20:41:53 2020

@author: keval
"""

import pandas as pd
import pickle

# Load from file
with open(r"C:\Users\keval\OneDrive\Desktop\Cloud_Counselage\New folder\domain_model.pkl", 'rb') as file:
    pickle_event_domain_model = pickle.load(file)

# Load from file
with open(r"C:\Users\keval\OneDrive\Desktop\Cloud_Counselage\New folder\event_model.pkl", 'rb') as file:
    pickle_event_type_model = pickle.load(file)

employee = pd.read_csv(r'C:\Users\keval\OneDrive\Desktop\Cloud_Counselage\New folder\CCMLEmployeeData.csv')   
sentences = pd.read_csv(r'C:\Users\keval\OneDrive\Desktop\Cloud_Counselage\New folder\testing.csv', names=['Event Title'], encoding= 'unicode_escape')  ##Ensure you have uploaded Input.csv in same folder; else result will be of default Input.csv provided
employee['Domain'] = employee['domain'].apply(lambda x:x.replace(' ',''))



domain = employee['domain'].tolist()
event1= employee['event1'].tolist()
event2 = employee['event2'].tolist()
name = employee['name'].tolist()


#Output DataFrame which will later be converted to XLS format
Dataframe=pd.DataFrame({'Event Title': [],'Recommended Employees': []})


for index, row in sentences.iterrows():
    Event_Title = row['Event Title']
    predicted_domain = list(pickle_event_domain_model.predict([Event_Title]))
    domain_list = predicted_domain[0].split(' ') 
    if domain_list == list():
      domain_list = ['Other']
    predicted_event = list(pickle_event_type_model.predict([Event_Title]))
    event_list = predicted_event[0].split(' ')
    if event_list == list():
      event_list = ['Webinars']
    
    

    lst = []
    for i in range(len(domain_list)):
        string = domain_list[i]
        string = string.lower()
        lst.append(string)
    lst1 = []
    for i in range(len(event_list)):
        string = event_list[i]
        string = string.lower()
        lst1.append(string)
    domain_list = lst
    event_list = lst1
    
    
    
    
    ## String Matching Algorithm to check which Employees have interests as predicted Domain & Event type
    for domain in domain_list:
      for event in event_list:
        relevant_employees = list(employee.loc[(employee['domain'] == domain) & ( (employee['event1'] == event) | (employee['event2'] == event))]['name'])
        single_event_result = ','.join(relevant_employees)
        if len(single_event_result) == 0:
            single_event_result = 'Not predicted'
        Dataframe = Dataframe.append({'Event Title': Event_Title,'Recommended Employees': single_event_result}, ignore_index=True)
    
    
    






Dataframe["Event Title"] =Dataframe.drop_duplicates(subset = ["Event Title",'Recommended Employees'], keep='first', inplace=False)
Dataframe= Dataframe.dropna()
Dataframe['Recommended Employees']
# Generating Output Excel file
Dataframe.to_excel('Output.xlsx', sheet_name='Output', index=False)

