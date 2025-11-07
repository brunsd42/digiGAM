#!/usr/bin/env python
# coding: utf-8

# additional tests
# - [ ]  list of combinations with less than 52 weeks
# - [ ]  repeated reach values for several weeks
# - [ ]  sudden increases / drops
# - [ ]  join with last years digi GAM and see step changes

# In[6]:


import pandas as pd 

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from tqdm import tqdm 


# In[3]:


digi_data = "../../Digital Data 2025/Digital Data 2025 - Weekly Combined.csv"
digi_df = pd.read_csv(digi_data)


# In[4]:


digi_df.sample()


# In[9]:


group


# In[12]:


pdf_filename = "graphs.pdf"

for (place, service, platform), group in tqdm(digi_df.groupby(['ServiceID', 'PlatformID', 'PlaceID'])):
    
    plt.figure(figsize=(8, 6))
    plt.plot(group['Week Number'], group['Reach'], marker='o', linestyle='-')
    plt.xlabel("Weeks")
    plt.ylabel("Reach")
    plt.title(f"Platform {platform} | Service {service} | Country {place}")
    plt.xticks(rotation=45)
    
    # Save the current figure to the PDF
    plt.savefig(f"../graphs/{platform}_{service}_{place}_{pdf_filename}", transparent=True)
    plt.close()


# In[ ]:




