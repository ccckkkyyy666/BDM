## Getting Started

Follow these steps to get started with the PPO Switch project:

### Prerequisites
- Python 3.10

### Installation
1. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
2. Install FinRL for Windows:
   ```bash
    git clone https://github.com/AI4Finance-Foundation/FinRL.git
    cd FinRL
    pip install .
    ```

### Running the Test
1. Open the `test.py` script and locate the following lines:
    ```python
   TRADE_START_DATE = 'YOUR_TRADE_START_DATE'
   TRADE_END_DATE = 'YOUR_TRADE_END_DATE'
   FILE_PATH = 'YOUR_TEST_DATA_FILE_PATH'
   ```
   Replace the placeholders with the actual information of your test data file.

2. Run the test using the following command:
    ```bash
    python test.py
    ```