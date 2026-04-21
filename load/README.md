# Data Loader Utility

This is a lightweight Python utility designed to load datasets dynamically based on environment variables. It supports CSV and JSON formats using the pandas library.

---

### Prerequisites

Before running the script, ensure you have the necessary dependencies installed:

```bash
pip install pandas python-dotenv
```

### Configuration

The script uses a .env file to manage file paths and types securely. Create a file named .env in your root directory and add the following variables:

```text
FILE_PATH="path/to/your/data_file.csv"
FILE_TYPE="csv"
```

---

### Functionality

The load_data() function performs the following steps:

1.  Loads environment variables from the .env file.
2.  Retrieves the file path and file type configuration.
3.  Determines the appropriate pandas reading method (read_csv or read_json).
4.  Returns a pandas DataFrame containing the loaded data.

### Usage Example

To use this functionality in your project:

```python
from loader import load_data

# Load the data into a pandas DataFrame
df = load_data()

# Display the first few rows
print(df.head())
```

---

> Note: Ensure that the FILE_TYPE variable exactly matches the format of the file located at FILE_PATH.