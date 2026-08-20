#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
experimentos_cat_states.py
==========================
Experimentos com estados do gato (cat states) para o UCGE-DC.

Estados do gato: (|x> + |x_bar>) / sqrt(2), onde x_bar é o complemento bit a bit de x.

O GHZ é um caso especial x = 00...0, x_bar = 11...1.

Resultados são salvos em: experimentos_cat_states_<data>.csv

Casos cobertos:
  1. GHZ para n=3..7 (referência)
  2. Estados do gato generalizados para n=4:
     - |0001> + |1110>  (1 bit vs 3 bits em |1>)
     - |0011> + |1100>  (2 bits vs 2 bits, paridade par)
     - |0101> + |1010>  (alternado)
     - |0110> + |1001>  (alternado 2)
  3. Estados do gato para n=6: - vários padrões de x
  4. Comparação: quantos níveis têm pares dc em cada caso
"""

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pnm"))

import numpy as np
from datetime import date
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector

from qclib.state_preparation.ucge import UCGEInitialize
from ucge_dc import UCGEDCInitialize

BASIS = ["cx", "u"]
OUTPUT_FILE = f"experimentos_cat_states_{date.today().strftime('%d%b%y').lower()}.csv"


def _build_circuit(state, cls):
    n = len(state).bit_length() - 1
    qc = QuantumCircuit(n)
    cls.initialize(qc, state)
    return qc


def _statevector(qc):
    return np.array(Statevector.from_instruction(qc))


def cnot_and_depth_counts(qc):
    t = transpile(qc, basis_gates=BASIS, optimization_level=0)
    return t.count_ops().get("cx", 0), t.depth()


def fidelity(psi, sv):
    return float(abs(np.dot(np.conj(psi), sv)) ** 2)


def evaluate(state):
    state = state / np.linalg.norm(state)
    results = {}
    for name, cls in [("ucge", UCGEInitialize), ("ucge_dc", UCGEDCInitialize)]:
        qc  = _build_circuit(state, cls)
        sv  = _statevector(qc)
        fid = fidelity(state, sv)
        cx, dep = cnot_and_depth_counts(qc)
        results[name] = {"fidelity": fid, "cnots": cx, "depth": dep}
    return results


def make_cat_state(n, x):
    """
    Cria o estado do gato (|x> + |x_bar>) / sqrt(2) para n qubits, onde
    x é um inteiro representando o índice do primeiro estado da base.
    x_bar é o complemento a um de x com n bits (flip bit a bit).
    """
    N = 2 ** n
    x_bar = (~x) & (N - 1)  # complemento a um de x com n bits
    psi = np.zeros(N)
    psi[x]     = 1.0 / np.sqrt(2)
    psi[x_bar] = 1.0 / np.sqrt(2)
    return psi, x, x_bar


def count_dc_pairs(psi):
    """
    Conta quantos pares (k) têm norma zero no nível mais baixo = Número de don't cares.
    """
    n = len(psi).bit_length() - 1
    dc_count = 0
    for k in range(len(psi) // 2):
        if abs(psi[2*k])**2 + abs(psi[2*k+1])**2 < 1e-10:
            dc_count += 1
    return dc_count


def print_row(label, r, failures):
    ucge = r["ucge"]
    dc   = r["ucge_dc"]
    dcx  = ucge["cnots"] - dc["cnots"]
    ok_u = "OK" if ucge["fidelity"] > 1 - 1e-6 else "!!"
    ok_d = "OK" if dc["fidelity"]   > 1 - 1e-6 else "!!"
    diff = f"+{dcx}" if dcx > 0 else (f"{dcx}" if dcx < 0 else "=")
    print(
        f"  {label:<55}"
        f"  {ok_u} {ucge['cnots']:>4} {ucge['depth']:>4}"
        f"  {ok_d} {dc['cnots']:>4} {dc['depth']:>4}"
        f"  {diff:>5}"
    )
    if ucge["fidelity"] < 1 - 1e-6:
        failures.append(f"UCGE fidelidade baixa: {label}")
    if dc["fidelity"] < 1 - 1e-6:
        failures.append(f"UCGE-DC fidelidade baixa: {label}")


def run_experiments():
    w = 100
    print("=" * w)
    print("  EXPERIMENTOS UCGE-DC — ESTADOS DO GATO (CAT STATES)")
    print("=" * w)
    print(f"  {'Estado':<55}  {'UCGE':^12}  {'DC':^12}  {'Dcx':>5}")
    print(f"  {'':55}  {'fid cx dep':^12}  {'fid cx dep':^12}")
    print("-" * w)

    failures = []
    csv_rows = []
    cases    = []

    # GHZ para n=3..7 (referência) 
    print("\n  -- GHZ (referência) --")
    for n in range(3, 8):
        psi, x, x_bar = make_cat_state(n, 0)
        label = (f"GHZ n={n}: "
                 f"|{'0'*n}>+|{'1'*n}>")
        cases.append((label, psi))

    # Estados do gato para n=4 
    print("\n  -- Cat states n=4 --")
    n = 4
    N = 2**n
    # todos os pares (x, x_bar) únicos para n=4
    seen = set()
    for x in range(N):
        x_bar = (~x) & (N-1)
        pair = tuple(sorted([x, x_bar]))
        if pair not in seen and x != x_bar:
            seen.add(pair)
            psi, _, _ = make_cat_state(n, x)
            dc_pairs  = count_dc_pairs(psi)
            label = (f"cat n=4: |{format(x,f'0{n}b')}>+|{format(x_bar,f'0{n}b')}>"
                     f" (dc_pairs={dc_pairs})")
            cases.append((label, psi))

    # Estados do gato para n=6 
    print("\n  -- Cat states n=6 --")
    n = 6
    N = 2**n
    # selecionar casos representativos
    xs_n6 = [
        0b000001,  # |000001> + |111110>
        0b000011,  # |000011> + |111100>
        0b000111,  # |000111> + |111000>
        0b001111,  # |001111> + |110000>
        0b010101,  # |010101> + |101010>
        0b011001,  # |011001> + |100110>
        0b001001,  # |001001> + |110110>
    ]
    for x in xs_n6:
        x_bar = (~x) & (N-1)
        psi, _, _ = make_cat_state(n, x)
        dc_pairs  = count_dc_pairs(psi)
        label = (f"cat n=6: |{format(x,f'0{n}b')}>+|{format(x_bar,f'0{n}b')}>"
                 f" (dc_pairs={dc_pairs})")
        cases.append((label, psi))

    # Rodar todos os casos 
    for label, state in cases:
        try:
            r = evaluate(state)
            print_row(label, r, failures)
            csv_rows.append([
                label,
                f"{r['ucge']['fidelity']:.10f}",
                r["ucge"]["cnots"],
                r["ucge"]["depth"],
                f"{r['ucge_dc']['fidelity']:.10f}",
                r["ucge_dc"]["cnots"],
                r["ucge_dc"]["depth"],
                r["ucge"]["cnots"] - r["ucge_dc"]["cnots"],
            ])
        except Exception as e:
            print(f"  {'ERRO':<55}  {label}: {e}")
            failures.append(f"ERRO: {label}: {e}")
            csv_rows.append([label, "ERRO", str(e), "", "", "", "", ""])

    print("=" * w)
    if failures:
        print("\nFALHAS:")
        for f in failures:
            print(f"   {f}")
    else:
        print("\n  Todos os testes de fidelidade passaram OK")

    # Resumo dos resultados
    deltas = [row[7] for row in csv_rows
              if isinstance(row[7], (int, float))]
    if deltas:
        wins   = sum(1 for d in deltas if d > 0)
        ties   = sum(1 for d in deltas if d == 0)
        losses = sum(1 for d in deltas if d < 0)
        total  = len(deltas)
        print(f"\n  Resumo: {total} casos")
        print(f"    UCGE-DC vence : {wins:>3} ({100*wins/total:.1f}%)")
        print(f"    Empate        : {ties:>3} ({100*ties/total:.1f}%)")
        print(f"    UCGE-DC perde : {losses:>3} ({100*losses/total:.1f}%)")
        if wins > 0:
            win_deltas = [d for d in deltas if d > 0]
            print(f"    Ganho médio   : {sum(win_deltas)/len(win_deltas):.1f} CNOTs")
            print(f"    Ganho máximo  : {max(win_deltas)} CNOTs")

    # Salvar em arquivo CSV
    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["estado",
                         "fid_ucge", "cx_ucge", "dep_ucge",
                         "fid_dc",   "cx_dc",   "dep_dc",
                         "delta_cx"])
        writer.writerows(csv_rows)
    print(f"\n  -> {OUTPUT_FILE} salvo.")


if __name__ == "__main__":
    run_experiments()
