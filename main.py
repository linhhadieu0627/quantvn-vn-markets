
import os
import pandas as pd
from dotenv import load_dotenv
from quantvn.vn.data.utils import client
from quantvn.vn.data import get_stock_hist

# 1. Gọi API Key bảo mật từ file .env
load_dotenv()
api_key = os.getenv("QUANT_API_KEY")
client(apikey=api_key)

# 2. Test lấy dữ liệu lịch sử mã VIC
df = get_stock_hist("VIC", resolution="1H")

# 3. In kết quả ra màn hình
print(df.tail())
