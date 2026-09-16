# Day 7: Dataset Receipt & Initial Inspection

## 📋 Deliverables Overview
* **Raw-Data Immutable Archive Setup**
* **Initial Inspection Validation Notebook Execution**
* **Preliminary Dataset Health Report**

---

## 🛠️ Google Colab Interactive Notebook Code Execution

### Cell Execution: Upload Project Zip Package
```python
from google.colab import files
uploaded = files.upload()
```
**Console Output Log:**
```text
Upload widget is only available when the cell has been executed in the current browser session. Please rerun this cell to enable.
Saving UWOC_useing_photoresistor-main.zip to UWOC_useing_photoresistor-main (1).zip
```

---

### Cell Execution: Extract Compressed Target Archive
```python
import zipfile
import io

# Assuming the uploaded file is in the 'uploaded' dictionary
# The file was uploaded as 'UWOC_useing_photoresistor-main (1).zip'
zip_file_name = 'UWOC_useing_photoresistor-main (1).zip'

# Unzip the file
with zipfile.ZipFile(io.BytesIO(uploaded[zip_file_name]), 'r') as z:
    z.extractall('.')
print(f"Extracted {zip_file_name} contents.")
```
**Console Output Log:**
```text
Extracted UWOC_useing_photoresistor-main (1).zip contents.
```

---

### Cell Execution: Load Matrix and Verify Dataset Geometry Dimensions
```python
import pandas as pd

df = pd.read_csv('UWOC_useing_photoresistor-main/anaotherGUI/1-2NTU.csv')
print(df.shape)
df.head()
```
**Console Output Log:**
```text
(8512, 3)

   Reading_Index  Analog_Voltage_mV  target_bit
0              1               2340           1
1              2               2345           1
2              3               1120           0
3              4               1115           0
4              5               2338           1
```
