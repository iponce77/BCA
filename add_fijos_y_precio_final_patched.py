
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
add_fijos_y_precio_final.py
---------------------------
- Suma 'valor_eur' del CSV de fijos.
- Añade columna fija con ese total (por defecto: fijos_es_eur).
- Calcula columna de 'precio final' = winning_bid + transporte + commission_total_eur + fijos.
  * Transporte: intenta 'transport_price_eur' y si no existe usa 'transport_eur'.

Uso:
  python add_fijos_y_precio_final.py --fijos bca_otros_gastos_es.csv     --excel-in fichas_vehiculos.xlsx     --excel-out fichas_vehiculos_final.xlsx     --col-fijos fijos_es_eur     --col-final precio_final_eur     --transport-col transport_price_eur

Notas:
- Si no indicas --excel-out, sobrescribe el de entrada.
- Si omites --transport-col, el script autodetecta: primero 'transport_price_eur', luego 'transport_eur'.
- Convierte numéricos con to_numeric y rellena NaN con 0 para evitar errores de suma.
"""

import argparse
import sys
import pandas as pd

def _to_num(s):
    return pd.to_numeric(s, errors='coerce')

def main():
    ap = argparse.ArgumentParser(description='Añadir fijos y calcular precio final')
    ap.add_argument('--fijos', required=True, help='CSV con columna valor_eur')
    ap.add_argument('--excel-in', required=True, help='Excel de entrada')
    ap.add_argument('--excel-out', help='Excel de salida (si no, se sobrescribe el de entrada)')
    ap.add_argument('--col-fijos', default='fijos_es_eur', help='Nombre columna de fijos (default: fijos_es_eur)')
    ap.add_argument('--col-final', default='precio_final_eur', help='Nombre columna precio final (default: precio_final_eur)')
    ap.add_argument('--transport-col', help='Nombre de columna de transporte (si no, auto: transport_price_eur/transport_eur/transport_eur_fallback)')
    ap.add_argument('--base-col', default='winning_bid', help='Columna base de precio (default: winning_bid)')
    ap.add_argument('--require-components', default='', help='Lista separada por comas de componentes obligatorios: transport,commissions,fixed')
    args = ap.parse_args()

    # 1) Leer fijos
    df_fijos = pd.read_csv(args.fijos)
    if 'valor_eur' not in df_fijos.columns:
        sys.exit("ERROR: el CSV de fijos debe tener columna 'valor_eur'.")
    total_fijos = float(_to_num(df_fijos['valor_eur']).fillna(0).sum())

    # 2) Leer Excel
    df = pd.read_excel(args.excel_in)

    # 3) Determinar columnas
    # base price
    base_col = args.base_col
    if base_col not in df.columns:
        sys.exit(f"ERROR: no encuentro columna base '{base_col}' en el Excel de entrada.")
    # transport
    transport_col = args.transport_col
    if not transport_col:
        if 'transport_price_eur' in df.columns:
            transport_col = 'transport_price_eur'
        elif 'transport_eur' in df.columns:
            transport_col = 'transport_eur'
        elif 'transport_eur_fallback' in df.columns:
            transport_col = 'transport_eur_fallback'
        else:
            transport_col = None
    # commission
    comm_col = 'commission_total_eur' if 'commission_total_eur' in df.columns else None
    if comm_col is None and 'bca_commission_eur' in df.columns:
        comm_col = 'bca_commission_eur'  # fallback

    # 3b) Requeridos
    required = set([x.strip() for x in (args.require_components or '').split(',') if x.strip()])
    def _require(name, ok):
        if name in required and not ok:
            sys.exit(f"ERROR: componente requerido ausente: {name}")
    _require('transport', transport_col is not None and transport_col in df.columns)
    _require('commissions', comm_col is not None and comm_col in df.columns)
    _require('fixed', True)  # se añade siempre más abajo

    # 4) Añadir columna de fijos
    df[args.col_fijos] = total_fijos

    # 5) Calcular precio final
    base = _to_num(df[base_col]).fillna(0)
    transport = _to_num(df[transport_col]).fillna(0) if transport_col and transport_col in df.columns else 0
    commission = _to_num(df[comm_col]).fillna(0) if comm_col and comm_col in df.columns else 0
    fijos = _to_num(df[args.col_fijos]).fillna(0)

    df[args.col_final] = base + transport + commission + fijos

    # 6) Guardar
    out_path = args.excel_out or args.excel_in
    df.to_excel(out_path, index=False)
    print(f"OK -> {out_path} | {args.col_fijos}={total_fijos:.2f} | base='{base_col}' | transporte='{transport_col or '0'}' | comisiones='{comm_col or '0'}' | columna final='{args.col_final}'")

if __name__ == '__main__':
    main()
