📦 Beta & Alpha Diversity Analysis App 使用說明（跨平台）

===========================================
✅ 1. 安裝環境（Windows / macOS / Linux）
===========================================

請先安裝 Python（建議版本 >= 3.9）
官網下載：https://www.python.org/downloads/

然後使用終端機執行：

pip install -r requirements.txt

（這行會自動安裝本系統所需套件）

===========================================
▶️ 2. 啟動系統
===========================================

在解壓後的資料夾中，執行以下指令：

Windows 用戶：
----------------
直接雙擊：
    start_gui.bat

或在命令列輸入：
    streamlit run app.py


macOS / Linux 用戶：
----------------
打開 Terminal，執行：
    streamlit run app.py

===========================================
🧭 3. WSL (Linux) 進入 Windows 桌面資料夾的方式
===========================================

如果你使用的是 WSL，想要從 Linux 存取 Windows 桌面，請使用：

cd /mnt/c/Users/<你的帳號名稱>/Desktop

例如：
cd /mnt/c/Users/youmin/Desktop

注意：如果是中文作業系統，資料夾可能叫「桌面」，則請使用：
cd /mnt/c/Users/<你的帳號>/桌面

===========================================
📁 4. 上傳資料格式說明
===========================================

1. Beta Diversity 分析需上傳：
   - sample sheet（CSV，包含 Sample 與 Group 欄位）
   - braycurtis 距離矩陣（TSV）

2. Alpha Diversity 分析需上傳：
   - CSV 檔，包含欄位：Sample、Group、observed_features、shannon_entropy

===========================================
🎯 5. 分析輸出
===========================================

- 自動偵測組別與樣本數
- Beta：支援 PERMANOVA / ANOSIM 比較
- Alpha：兩兩組別進行 Mann-Whitney U 檢定
- 可設定 permutation 次數與隨機種子

如需圖表輸出、報告生成功能，歡迎持續擴充！
