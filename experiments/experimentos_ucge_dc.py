#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
experimentos_ucge_dc.py
=======================
Experimentos estendidos para o UCGE-DC, cobrindo casos complexos
adicionais não contemplados no test_ucge_dc.py original.

Resultados salvos em:
  - experimentos_ucge_dc_<timestamp>.csv

Casos cobertos:
  1. Esparso complexo original (5 casos do test_ucge_dc.py)
  2. Casos adicionais:
     - n=6 e n=7 com amplitudes complexas esparsas
     - Denso complexo (deve empatar com UCGE)
     - Padrão alternado de fases (variante complexa do esparso real)
     - Bloco esparso repetido para n=6
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
OUTPUT_FILE = f"experimentos_ucge_dc_{date.today().strftime('%d%b%y').lower()}.csv"


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


def _print_row(label, r, failures):
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
    ph = np.exp
    a, b, c, d = 0.3, 0.7, 0.5, 0.4

    cases = [
        # ── Casos originais do test_ucge_dc.py ────────────────────────────────
        ("orig | n=3 cplx dc+irmao fase",
         np.array([0, 0,
                   a*ph(1j*0.3), b*ph(1j*0.7),
                   c*ph(1j*1.1), d*ph(1j*0.2),
                   c*ph(1j*1.1), d*ph(1j*0.2)])),

        ("orig | n=4 cplx bloco ativo fases distintas",
         np.array([0]*8 +
                  [a*ph(1j*0.5), b*ph(1j*1.2),
                   c*ph(1j*0.8), d*ph(1j*2.1),
                   c*ph(1j*0.8), d*ph(1j*2.1),
                   0, 0])),

        ("orig | n=4 cplx 3 spikes isolados",
         np.array([0.6+0.3j if i == 5 else
                   0.2-0.5j if i == 9 else
                   0.4+0.1j if i == 13 else 0.0
                   for i in range(16)])),

        ("orig | n=4 cplx 4 spikes fases distintas",
         np.array([0.5*ph(1j*0.3) if i == 2 else
                   0.5*ph(1j*1.1) if i == 3 else
                   0.5*ph(1j*2.0) if i == 8 else
                   0.5*ph(1j*0.7) if i == 14 else 0.0
                   for i in range(16)])),

        ("orig | n=5 cplx bloco esparso repetido",
         np.concatenate([
             np.zeros(16),
             np.zeros(6),
             np.array([a*ph(1j*0.4), b*ph(1j*0.9)]),
             np.array([c*ph(1j*1.5), d*ph(1j*0.6),
                       c*ph(1j*1.5), d*ph(1j*0.6),
                       c*ph(1j*1.5), d*ph(1j*0.6),
                       c*ph(1j*1.5), d*ph(1j*0.6)]),
         ])),

        # ── Casos adicionais: escalonamento para n=6 e n=7 ────────────────────
        ("novo | n=6 cplx bloco de zeros na primeira metade",
         np.concatenate([
             np.zeros(32),
             np.array([a*ph(1j*0.3), b*ph(1j*0.7),
                       c*ph(1j*1.1), d*ph(1j*0.2),
                       c*ph(1j*1.1), d*ph(1j*0.2),
                       c*ph(1j*1.1), d*ph(1j*0.2),
                       c*ph(1j*1.1), d*ph(1j*0.2),
                       a*ph(1j*0.5), b*ph(1j*1.2),
                       c*ph(1j*0.8), d*ph(1j*2.1),
                       c*ph(1j*0.8), d*ph(1j*2.1),
                       c*ph(1j*0.8), d*ph(1j*2.1),
                       c*ph(1j*0.8), d*ph(1j*2.1),
                       a*ph(1j*0.3), b*ph(1j*0.7),
                       c*ph(1j*1.1), d*ph(1j*0.2),
                       c*ph(1j*1.1), d*ph(1j*0.2),
                       c*ph(1j*1.1), d*ph(1j*0.2),
                       c*ph(1j*1.1), d*ph(1j*0.2),
                       a*ph(1j*0.5), b*ph(1j*1.2)]),
         ])),

        ("novo | n=7 cplx bloco de zeros na primeira metade",
         np.concatenate([
             np.zeros(64),
             np.array([a*ph(1j*0.3), b*ph(1j*0.7),
                       c*ph(1j*1.1), d*ph(1j*0.2)] * 16),
         ])),

        # ── Padrão alternado de fases (variante complexa do esparso real) ──────
        ("novo | n=3 cplx alternado [0,0,ae^i,be^i,ce^i,de^i,ce^i,de^i]",
         np.array([0, 0,
                   a*ph(1j*0.5), b*ph(1j*1.0),
                   c*ph(1j*1.5), d*ph(1j*2.0),
                   c*ph(1j*1.5), d*ph(1j*2.0)])),

        ("novo | n=4 cplx alternado [0]*8+[ae^i,be^i,ce^i,de^i,ce^i,de^i,ce^i,de^i]",
         np.array([0]*8 +
                  [a*ph(1j*0.5), b*ph(1j*1.0),
                   c*ph(1j*1.5), d*ph(1j*2.0),
                   c*ph(1j*1.5), d*ph(1j*2.0),
                   c*ph(1j*1.5), d*ph(1j*2.0)])),

        ("novo | n=5 cplx alternado familia Fn",
         np.concatenate([
             np.zeros(16),
             np.zeros(6),
             np.array([a*ph(1j*0.5), b*ph(1j*1.0)]),
             np.array([c*ph(1j*1.5), d*ph(1j*2.0),
                       c*ph(1j*1.5), d*ph(1j*2.0),
                       c*ph(1j*1.5), d*ph(1j*2.0),
                       c*ph(1j*1.5), d*ph(1j*2.0)]),
         ])),

        # ── Bloco esparso repetido para n=6 ───────────────────────────────────
        ("novo | n=6 cplx bloco esparso repetido",
         np.concatenate([
             np.zeros(32),
             np.zeros(14),
             np.array([a*ph(1j*0.4), b*ph(1j*0.9)]),
             np.array([c*ph(1j*1.5), d*ph(1j*0.6)] * 8),
         ])),

        # ── Denso complexo (deve empatar com UCGE) ─────────────────────────────
        ("novo | n=3 cplx denso aleatorio (deve empatar)",
         np.random.default_rng(10).random(8) *
         np.exp(1j * np.random.default_rng(11).uniform(0, 2*np.pi, 8))),

        ("novo | n=4 cplx denso aleatorio (deve empatar)",
         np.random.default_rng(12).random(16) *
         np.exp(1j * np.random.default_rng(13).uniform(0, 2*np.pi, 16))),

        ("novo | n=4 cplx denso fases uniformes (deve empatar)",
         np.array([np.exp(1j * k * np.pi / 8) / 4.0 for k in range(16)])),

        # ── Spikes isolados para n=5 e n=6 ────────────────────────────────────
        ("novo | n=5 cplx 4 spikes isolados fases distintas",
         np.array([0.5*ph(1j*0.3) if i == 3  else
                   0.5*ph(1j*1.1) if i == 9  else
                   0.5*ph(1j*2.0) if i == 17 else
                   0.5*ph(1j*0.7) if i == 25 else 0.0
                   for i in range(32)])),

        ("novo | n=6 cplx 5 spikes isolados fases distintas",
         np.array([0.45*ph(1j*0.3) if i == 5  else
                   0.45*ph(1j*1.1) if i == 15 else
                   0.45*ph(1j*2.0) if i == 27 else
                   0.45*ph(1j*0.7) if i == 40 else
                   0.45*ph(1j*1.5) if i == 55 else 0.0
                   for i in range(64)])),
    ]

    w = 100
    print("=" * w)
    print("  EXPERIMENTOS UCGE-DC — CASOS COMPLEXOS ESTENDIDOS")
    print("=" * w)
    print(f"  {'Estado':<55}  {'UCGE':^12}  {'DC':^12}  {'Dcx':>5}")
    print(f"  {'':55}  {'fid cx dep':^12}  {'fid cx dep':^12}")
    print("-" * w)

    failures = []
    csv_rows = []

    for label, state in cases:
        try:
            r = evaluate(state)
            _print_row(label, r, failures)
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

    # Resumo
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
