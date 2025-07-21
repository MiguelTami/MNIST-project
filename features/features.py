from features import extractor
import pandas as pd


def generar_features(df_imagenes):
    feature_list = []

    for _, row in df_imagenes.iterrows():
        img = row.values.reshape(28, 28)

        features = [
            extractor.intensidad_promedio(img),
            extractor.varianza_intensidad(img),
            extractor.densidad_superior(img),
            extractor.simetria_vertical(img),
            extractor.centro_masa_vertical(img),
            extractor.simetria_horizontal(img)
        ]

        feature_list.append(features)

        columnas = [
        "intensidad_promedio",
        "varianza",
        "densidad_superior",
        "simetria_vertical",
        "centro_masa_vertical",
        "simetria_horizontal"
    ]

    
    return pd.DataFrame(feature_list, columns=columnas)