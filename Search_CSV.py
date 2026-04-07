import os
import glob

print("Searching for clean*.csv files...")
for root, dirs, files in os.walk(r"\\wsl.localhost\Ubuntu\home\canal\github\treinamento-dados"):
    for file in files:
        if "clean" in file and file.endswith(".csv"):
            print(os.path.join(root, file))
