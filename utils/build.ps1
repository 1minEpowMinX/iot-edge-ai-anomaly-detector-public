pyinstaller --noconfirm --clean --onedir --name iot-edge-ai-anomaly-detector-win64 `
--version-file ".\resources\version_info.txt" `
--collect-all torch `
--collect-submodules sklearn `
--exclude-module tkinter `
--exclude-module tensorflow `
--exclude-module PyQt5 --exclude-module PyQt6 `
--exclude-module PySide2 --exclude-module PySide6 `
--exclude-module IPython --exclude-module pytest --exclude-module notebook `
main.py