import pandas as pd
import os

output_files = [
    "data/1erTrimestre/Claudia.csv", "data/1erTrimestre/Maynez.csv", "data/1erTrimestre/Xochitl.csv",
    "data/2doTrimestre/Claudia.csv", "data/2doTrimestre/Maynez.csv", "data/2doTrimestre/Xochitl.csv"]

target_files = [
    ["Clasificacion/Enero/Claudia.csv", "Clasificacion/Febrero/Claudia.csv", "Clasificacion/Marzo/Claudia.csv"], # 1erTrimestre/Claudia.csv
    ["Clasificacion/Enero/Maynez.csv", "Clasificacion/Febrero/Maynez.csv", "Clasificacion/Marzo/Maynez.csv"], # 1erTrimestre/Maynez.csv
    ["Clasificacion/Enero/Xochitl.csv", "Clasificacion/Febrero/Xochitl.csv", "Clasificacion/Marzo/Xochitl.csv"], # 1erTrimestre/Xochitl.csv
    ["Clasificacion/Abril/Claudia.csv", "Clasificacion/Mayo/Claudia.csv"], # 2doTrimestre/Claudia.csv
    ["Clasificacion/Abril/Maynez.csv", "Clasificacion/Mayo/Maynez.csv"], # 2doTrimestre/Maynez.csv
    ["Clasificacion/Abril/Xochitl.csv", "Clasificacion/Mayo/Xochitl.csv"], # 2doTrimestre/Xochitl.csv
]

# Join csv
for (index, row) in enumerate(target_files):
    output_file = []
    for file in row:
        df = pd.read_csv(file, index_col=0)
        output_file.append(df)
    
    # Concat in a dataframe
    frame = pd.concat(output_file, axis=0)
    print(len(frame))

    # Create dir if dont exists
    dir = os.path.dirname(output_files[index])
    if not os.path.exists(dir):
        os.makedirs(dir)

    # Save file
    frame.to_csv(output_files[index], index=False)