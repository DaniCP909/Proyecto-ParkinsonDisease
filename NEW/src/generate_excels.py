import pandas as pd
from pathlib import Path
import numpy as np
import os

BASE_PATH = "/home/dcorredor/github/Proyecto-ParkinsonDisease/NEW"
DCORR_BASE_PATH = "/home/dcorredor/github/Proyecto-ParkinsonDisease/NEW"
rel_path = "../analysis/2/tasks_2_CnnOnly_train_2025-12-15 12:37:35.443871.csv"
ANALYSIS_REL_DIR = "analysis"
ANALYSIS_DIR = "analysis"

def make_image_link(rel_path):
    full_path = os.path.join(BASE_PATH, rel_path)
    return f'=HYPERLINK("file://{full_path}","Ver imagen")'

def calc_excel_metrics(origin_file, destination_file):
    df = pd.read_csv(origin_file)
    df["healthy_neur_confidence"] = 1 - df["park_neur confidence"]
    df["error"] = np.where(
        df["prediction"] != df["target"],
        "X",
        ""
    )
    df["pred_ajustada"] = [
        f"=IF(D{i+2}>=config!$B$2,1,0)"
        for i in range(len(df))
    ]
    df_config = pd.DataFrame(
        {
            "Param": ["Umbral"],
            "Valor": [0.5]
        }
    )
    df["FP"] = [
        f"=IF(AND(G{i+2}=0,C{i+2}=1),1,0)"
        for i in range(len(df))
    ]

    df["FN"] = [
        f"=IF(AND(G{i+2}=1,C{i+2}=0),1,0)"
        for i in range(len(df))
    ]

    df["TP"] = [
        f"=IF(AND(G{i+2}=1,C{i+2}=1),1,0)"
        for i in range(len(df))
    ]

    df["TN"] = [
        f"=IF(AND(G{i+2}=0,C{i+2}=0),1,0)"
        for i in range(len(df))
    ]
    df["link"] = df["filename"].apply(make_image_link)
    df_metrics = pd.DataFrame({
        "Metrica": ["FP", "FN", "TP", "TN", "Total Global"],
        "Total": [
            "=SUM(datos!H:H)",
            "=SUM(datos!I:I)",
            "=SUM(datos!J:J)",
            "=SUM(datos!K:K)",
            "=SUM(B2:B5)"
        ],
        "Porcentaje": [
            "=(B2/B6)*100",
            "=(B3/B6)*100",
            "=(B4/B6)*100",
            "=(B5/B6)*100",
            1
        ]
    })
    df_hits = pd.DataFrame({
        "Tipo": ["Aciertos", "Errores"],
        "Total": [
            "=SUM(B4,B5)",
            "=SUM(B2,B3)"
        ],
        "Porcentaje": [
            "=(B9/B6)*100",
            "=(B10/B6)*100"
        ]
    })
    with pd.ExcelWriter(destination_file, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="datos", index=False)
        df_config.to_excel(writer, sheet_name="config", index=False)
        df_metrics.to_excel(writer, sheet_name="metricas", startrow=0, index=False)
        df_hits.to_excel(writer, sheet_name="metricas", startrow=7, index=False)

for root, dirs, files in os.walk(ANALYSIS_REL_DIR):
    print(f"{files}")
    for dir in dirs:
        dir_path = os.path.join(BASE_PATH, ANALYSIS_DIR, dir)
        filenames = os.listdir(dir_path)
        for file in filenames:
            origin_file = os.path.join(dir_path, file)

            no_extension_filename = origin_file[:-4]
            xlsx_file = no_extension_filename + ".xlsx"

            destination_file = os.path.join(dir_path, xlsx_file)
            calc_excel_metrics(origin_file, destination_file)
