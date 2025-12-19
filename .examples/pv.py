import os
from datetime import datetime, time, timedelta

import pandas as pd


# =====================================================================
# CONFIGURATION
# =====================================================================

# Caminho do arquivo Excel de entrada
excel_path = r".examples/sim/Simulation_WY_Fut_HP__PV5000-HB5000.xlsx"

# Nome da planilha (0 = primeira planilha)
sheet_name = 0

# Caminho do CSV de saída (por padrão, mesmo nome + sufixo)
output_csv_path = os.path.splitext(excel_path)[0] + "_PV_5min.csv"


# =====================================================================
# AUXILIARY FUNCTIONS
# =====================================================================

def build_datetime_sequence(date_time_col, base_step_minutes=15):
    """
    Reconstrói uma sequência de datetime com passo fixo (default: 15 min),
    a partir de uma coluna que mistura datetime, time e NaN.
    """
    dt_list = []
    current_date = None
    last_dt = None

    for value in date_time_col:
        if isinstance(value, datetime):
            # Valor já com data e hora
            current_date = value.date()
            last_dt = value
        elif isinstance(value, time):
            # Apenas hora; combinar com a data corrente
            if current_date is None:
                # fallback (não deve ocorrer, mas garante robustez)
                current_date = datetime(1900, 1, 1).date()
            last_dt = datetime.combine(current_date, value)
        else:
            # NaN ou outro tipo -> assumir próximo passo de tempo fixo
            if last_dt is None:
                # fallback inicial
                last_dt = datetime(1900, 1, 1, 0, 0)
            last_dt = last_dt + timedelta(minutes=base_step_minutes)

        dt_list.append(last_dt)

    return pd.to_datetime(dt_list)


# =====================================================================
# MAIN
# =====================================================================

def main():
    # 1) Ler Excel
    df = pd.read_excel(excel_path, sheet_name=sheet_name)

    # 2) Construir coluna de datetime consistente
    df["datetime"] = build_datetime_sequence(df["Date/Time"], base_step_minutes=15)

    # Garantir ordenação por tempo
    df = df.sort_values("datetime")

    # 3) Calcular PV em W (produced + charge - discharge) e cortar em zero
    pv_w = (
        df["produced_electricity_rate_W"]
        + df["charge_power_W"]
        - df["discharge_power_W"]
    )
    pv_w = pv_w.clip(lower=0)

    # 4) Converter para kW
    pv_kw = pv_w / 1000.0

    # 5) Montar série com datetime como índice
    pv = pd.DataFrame({"PV_kW": pv_kw.values}, index=df["datetime"])

    # Se houver timestamps duplicados, usar a média
    pv = pv.groupby(pv.index).mean()

    # 6) Gerar índice alvo de 5 em 5 minutos para 1 ano (365 dias)
    start = pv.index.min()
    n_steps_year = 365 * 24 * 60 // 5  # 365 dias, passo de 5 min

    # Usar "5min" em vez de "5T" (alias depreciado)
    full_index = pd.date_range(start=start, periods=n_steps_year, freq="5min")

    # 7) Reindexar e interpolar no tempo
    pv_5min = pv.reindex(full_index)

    # Interpolação linear no tempo
    pv_5min["PV_kW"] = pv_5min["PV_kW"].interpolate(method="time")

    # Nos extremos (onde não há dados para interpolar), preencher com 0
    pv_5min["PV_kW"] = pv_5min["PV_kW"].fillna(0.0)

    # 8) Salvar como CSV com as colunas datetime e PV_kW
    pv_5min = pv_5min.reset_index().rename(columns={"index": "datetime"})
    pv_5min.to_csv(output_csv_path, index=False)

    print(f"Arquivo salvo em: {output_csv_path}")


if __name__ == "__main__":
    main()
