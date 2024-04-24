import pandas as pd

from datetime import datetime


def create_dataframe(timestamps):
    ids = []
    datas_entrada = []
    horas_entrada = []
    datas_saida = []
    horas_saida = []

    # Iterating over timestamps to extract entry and exit date and time
    for id, times in timestamps.items():
        ids.append(id)
        if 'enter' in times:
            datas_entrada.append(times['enter'].date())
            horas_entrada.append(times['enter'].time())
        else:
            datas_entrada.append(None)
            horas_entrada.append(None)
        if 'exit' in times:
            datas_saida.append(times['exit'].date())
            horas_saida.append(times['exit'].time())
        else:
            datas_saida.append(None)
            horas_saida.append(None)

    # Creating the DataFrame
    df = pd.DataFrame({
        'ID': ids,
        'Data de entrada': datas_entrada,
        'Hora de entrada': horas_entrada,
        'Data de saída': datas_saida,
        'Hora de saída': horas_saida
    })
    return df

