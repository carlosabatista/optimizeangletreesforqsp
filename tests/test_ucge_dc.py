"""
test_ucge_dc.py
===============
Testes de fidelidade e contagem de CNOTs para:
  - UCGEInitialize   (UCGE original)
  - UCGEDCInitialize (UCGE com preenchimento ótimo de don't-cares)

Resultados salvos em:
  - resultados_concretos.csv
  - resultados_scaling.csv
  - resultados_benchmark.csv
"""

import csv
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector

from qclib.state_preparation.ucge import UCGEInitialize
from ucge_dc import UCGEDCInitialize


BASIS = ["cx", "u"]


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


def _fn_state(n, a=0.3, b=0.7, c=0.5, d=0.4):
    """
        Família Fn: primeira metade zeros, segunda metade com a seguinte estrutura:
        todos os elementos da primeira partição são de zeros, de modo a beneficiar
        o método. Os dois últimos elemento da primeira metade da segunda partição
        são as amplitudes a e b, sendo zero os anteriores, e a segunda metade da
        segunda partição alternam entre c e d. Significa que teremos muitos ângulos
        iguais no segundo quarto do estado. Tal configuração deve beneficiar nosso método
        tanto pelos dc's quanto pelos ângulos iguais.
    """
    N      = 2 ** n
    half   = N // 2
    psi    = np.zeros(N)
    rel_ab = half // 2 - 2
    psi[half + rel_ab]     = a
    psi[half + rel_ab + 1] = b
    for r in range(rel_ab + 2, half, 2):
        psi[half + r]     = c
        psi[half + r + 1] = d
    return psi / np.linalg.norm(psi)


def _print_row(label, r, failures):
    ucge = r["ucge"]
    dc   = r["ucge_dc"]
    dcx  = ucge["cnots"] - dc["cnots"]
    ok_u = "OK" if ucge["fidelity"] > 1 - 1e-6 else "!!"
    ok_d = "OK" if dc["fidelity"]   > 1 - 1e-6 else "!!"
    diff = f"+{dcx}" if dcx > 0 else (f"{dcx}" if dcx < 0 else "=")
    print(
        f"  {label:<50}"
        f"  {ok_u} {ucge['cnots']:>4} {ucge['depth']:>4}"
        f"  {ok_d} {dc['cnots']:>4} {dc['depth']:>4}"
        f"  {diff:>5}"
    )
    if ucge["fidelity"] < 1 - 1e-6:
        failures.append(f"UCGE fidelidade baixa: {label}")
    if dc["fidelity"] < 1 - 1e-6:
        failures.append(f"UCGE-DC fidelidade baixa: {label}")


# -- Casos concretos -- (Com valores numéricos fixos)

def test_concrete():
    a, b, c, d = 0.3, 0.7, 0.5, 0.4
    ph = np.exp

    cases = [
        # estados reais
        ("n=3 [0,0,a,b,c,d,c,d]",
         np.array([0, 0, a, b, c, d, c, d], dtype=float)),

        ("n=4 [0]*8+[0,0,a,b,c,d,c,d]",
         np.array([0]*8 + [0, 0, a, b, c, d, c, d], dtype=float)),

        ("n=5 familia Fn",
         _fn_state(5)),

        ("n=4 spike unico |0101>",
         np.array([1.0 if i == 5 else 0.0 for i in range(16)])),

        ("n=4 dois spikes |0000>+|1111>",
         np.array([1.0 if i in (0, 15) else 0.0 for i in range(16)])),

        ("n=4 quarteto ativo {8,9,10,11}",
         np.array([0]*8 + [a, b, c, d] + [0]*4, dtype=float)),

        ("n=4 bloco unico |10xx>",
         np.array([0]*8 + [a, b, c, d, 0, 0, 0, 0], dtype=float)),

        # estados complexos esparsos
        ("n=3 cplx dc+irmao fase [0,0,a*e^i,b*e^i,...]",
         np.array([0, 0,
                   a*ph(1j*0.3), b*ph(1j*0.7),
                   c*ph(1j*1.1), d*ph(1j*0.2),
                   c*ph(1j*1.1), d*ph(1j*0.2)])),

        ("n=4 cplx bloco ativo c/ fases distintas",
         np.array([0]*8 +
                  [a*ph(1j*0.5), b*ph(1j*1.2),
                   c*ph(1j*0.8), d*ph(1j*2.1),
                   c*ph(1j*0.8), d*ph(1j*2.1),
                   0, 0])),

        ("n=4 cplx 3 spikes isolados",
         np.array([0.6+0.3j if i == 5 else
                   0.2-0.5j if i == 9 else
                   0.4+0.1j if i == 13 else 0.0
                   for i in range(16)])),

        ("n=4 cplx 4 spikes fases distintas",
         np.array([0.5*ph(1j*0.3) if i == 2 else
                   0.5*ph(1j*1.1) if i == 3 else
                   0.5*ph(1j*2.0) if i == 8 else
                   0.5*ph(1j*0.7) if i == 14 else 0.0
                   for i in range(16)])),

        ("n=5 cplx bloco esparso repetido",
         np.concatenate([
             np.zeros(16),
             np.zeros(6),
             np.array([a*ph(1j*0.4), b*ph(1j*0.9)]),
             np.array([c*ph(1j*1.5), d*ph(1j*0.6),
                       c*ph(1j*1.5), d*ph(1j*0.6),
                       c*ph(1j*1.5), d*ph(1j*0.6),
                       c*ph(1j*1.5), d*ph(1j*0.6)]),
         ])),

        ("n=4 pior caso benchmark (deve empatar)",
         np.array([0.1552, 0, 0, 0, 0, 0, 0, 0.0089,
                   0.6145, 0, 0.4245, 0, 0.5366, 0, 0, 0.3607])),

        ("n=4 pior caso cplx (fases nos spikes)",
         np.array([(0.1552+0.05j) if i == 0 else
                   (0.0089-0.02j) if i == 7 else
                   (0.6145+0.1j)  if i == 8 else
                   (0.4245-0.15j) if i == 10 else
                   (0.5366+0.08j) if i == 12 else
                   (0.3607-0.05j) if i == 15 else 0.0
                   for i in range(16)])),

        # estados densos
        ("n=3 denso aleatorio real",
         np.random.default_rng(1).random(8)),

        ("n=4 denso aleatorio real",
         np.random.default_rng(2).random(16)),

        ("n=4 denso aleatorio complexo",
         np.random.default_rng(3).random(16) +
         1j * np.random.default_rng(4).random(16)),

        ("n=4 denso cplx fases uniformes",
         np.array([np.exp(1j * k * np.pi / 8) / 4.0 for k in range(16)])),
    ]

    w = 93
    print("=" * w)
    print("CASOS CONCRETOS")
    print("-" * w)
    print(f"  {'Estado':<50}  {'UCGE':^12}  {'DC':^12}  {'Dcx':>5}")
    print(f"  {'':50}  {'fid cx dep':^12}  {'fid cx dep':^12}")
    print("-" * w)

    failures  = []
    csv_rows  = []

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
            print(f"  {'ERRO':<50}  {label}: {e}")
            failures.append(f"ERRO: {label}: {e}")
            csv_rows.append([label, "ERRO", str(e), "", "", "", "", ""])

    print("=" * w)
    if failures:
        print("\nFALHAS:")
        for f in failures:
            print(f"   {f}")
    else:
        print("\n  Todos os testes de fidelidade passaram OK")
    print()

    with open("resultados_concretos.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["estado",
                         "fid_ucge", "cx_ucge", "dep_ucge",
                         "fid_dc",   "cx_dc",   "dep_dc",
                         "delta_cx"])
        writer.writerows(csv_rows)
    print("  -> resultados_concretos.csv salvo.")


# -- Escalonamento família Fn ---------------------------

def test_scaling():
    """Vantagem em CNOTs para a família Fn, n=3..7."""
    print("ESCALONAMENTO -- familia Fn")
    print("-" * 65)
    print(f"  {'n':>3}  {'N':>5}  {'nnz':>5}  "
          f"{'cx_ucge':>8}  {'cx_dc':>6}  {'D':>5}  "
          f"{'fid_ucge':>10}  {'fid_dc':>10}")
    print("  " + "-" * 62)

    csv_rows = []

    for n in range(3, 8):
        psi = _fn_state(n)
        nnz = int(np.sum(np.abs(psi) > 1e-10))
        try:
            r    = evaluate(psi)
            cx_u = r["ucge"]["cnots"]
            cx_d = r["ucge_dc"]["cnots"]
            dp_u = r["ucge"]["depth"]
            dp_d = r["ucge_dc"]["depth"]
            fi_u = r["ucge"]["fidelity"]
            fi_d = r["ucge_dc"]["fidelity"]
            ok_u = "OK" if fi_u > 1 - 1e-6 else "!!"
            ok_d = "OK" if fi_d > 1 - 1e-6 else "!!"
            print(f"  {n:>3}  {2**n:>5}  {nnz:>5}  "
                  f"{cx_u:>8}  {cx_d:>6}  {cx_u-cx_d:>+5}  "
                  f"{ok_u} {fi_u:.8f}  {ok_d} {fi_d:.8f}")
            csv_rows.append([n, 2**n, nnz,
                             cx_u, dp_u, cx_d, dp_d, cx_u - cx_d,
                             f"{fi_u:.10f}", f"{fi_d:.10f}"])
        except Exception as e:
            print(f"  {n:>3}  ERRO: {e}")
            csv_rows.append([n, 2**n, nnz, "ERRO", str(e), "", "", "", "", ""])

    print()

    with open("resultados_scaling.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["n", "N", "nnz",
                         "cx_ucge", "dep_ucge",
                         "cx_dc",   "dep_dc",
                         "delta_cx",
                         "fid_ucge", "fid_dc"])
        writer.writerows(csv_rows)
    print("  -> resultados_scaling.csv salvo.")
    print()


# -- Benchmark estatístico -------------------------------------

def test_benchmark(n_qubits=4, n_samples=500, seed=42):
    """
    Benchmark com estados esparsos aleatórios.
    Metade das amostras tem amplitudes reais, metade complexas.
    """
    rng = np.random.default_rng(seed)
    N   = 2 ** n_qubits

    wins = ties = losses = 0
    fid_fail_ucge = fid_fail_dc = 0
    delta_list = []
    best_win   = (None, 0, 0, 0)
    worst_loss = (None, 0, 0, 0)

    n_real    = 0
    n_complex = 0
    csv_rows  = []

    print(f"BENCHMARK -- n={n_qubits}, {n_samples} estados esparsos aleatorios")
    print(f"  (amostras pares: reais; impares: complexas)")
    print("-" * 60)

    for i in range(n_samples):
        nnz  = rng.integers(1, N)   # intervalo [1, N-1]
        idx  = rng.choice(N, nnz, replace=False)
        amps = rng.random(nnz) + 0.01

        if i % 2 == 0:
            tipo  = "real"
            state = np.zeros(N, dtype=float)
            n_real += 1
        else:
            tipo  = "complexo"
            amps  = amps * np.exp(1j * rng.uniform(0, 2*np.pi, nnz))
            state = np.zeros(N, dtype=complex)
            n_complex += 1

        for k, v in zip(idx, amps):
            state[k] = v
        state /= np.linalg.norm(state)

        try:
            r = evaluate(state)
        except Exception as e:
            print(f"  amostra {i} -- ERRO: {e}")
            csv_rows.append([i, tipo, nnz, "ERRO", str(e), "", "", "", "", ""])
            continue

        fi_u  = r["ucge"]["fidelity"]
        fi_d  = r["ucge_dc"]["fidelity"]
        cx_o  = r["ucge"]["cnots"]
        cx_d  = r["ucge_dc"]["cnots"]
        dp_o  = r["ucge"]["depth"]
        dp_d  = r["ucge_dc"]["depth"]
        delta = cx_o - cx_d

        if fi_u < 1 - 1e-6: fid_fail_ucge += 1
        if fi_d < 1 - 1e-6: fid_fail_dc   += 1

        delta_list.append(delta)

        if   delta > 0:  wins   += 1
        elif delta == 0: ties   += 1
        else:            losses += 1

        if delta > best_win[3]:
            best_win = (state.copy(), cx_o, cx_d, delta)
        if delta < worst_loss[3]:
            worst_loss = (state.copy(), cx_o, cx_d, delta)

        csv_rows.append([i, tipo, nnz,
                         cx_o, dp_o, cx_d, dp_d, delta,
                         f"{fi_u:.10f}", f"{fi_d:.10f}"])

    total = wins + ties + losses
    arr   = np.array(delta_list)

    print(f"  amostras processadas    : {total}")
    print(f"    reais                 : {n_real}")
    print(f"    complexas             : {n_complex}")
    print(f"  falhas fidelidade UCGE  : {fid_fail_ucge}")
    print(f"  falhas fidelidade DC    : {fid_fail_dc}")
    print()
    print(f"  UCGE-DC vence  : {wins:>5}  ({100*wins/total:.1f}%)")
    print(f"  empate         : {ties:>5}  ({100*ties/total:.1f}%)")
    print(f"  UCGE-DC perde  : {losses:>5}  ({100*losses/total:.1f}%)")
    print()
    print(f"  D medio        : {arr.mean():>+.2f} CNOTs")
    print(f"  D desvio-padrao: {arr.std():>+.2f} CNOTs")
    print(f"  D maximo (win) : {best_win[3]:>+d}"
          f"  (ucge={best_win[1]}, dc={best_win[2]})")
    print(f"  D maximo (loss): {worst_loss[3]:>+d}"
          f"  (ucge={worst_loss[1]}, dc={worst_loss[2]})")

    if best_win[0] is not None and best_win[3] > 0:
        nz = [(k, v) for k, v in enumerate(best_win[0]) if abs(v) > 1e-10]
        print(f"\n  Melhor caso (dc vence por {best_win[3]} CNOTs):")
        for k, v in nz[:8]:
            print(f"    |{k:0{n_qubits}b}> = {v:.4f}")
        if len(nz) > 8:
            print(f"    ... ({len(nz)} amplitudes nao-nulas)")

    if worst_loss[0] is not None and worst_loss[3] < 0:
        nz = [(k, v) for k, v in enumerate(worst_loss[0]) if abs(v) > 1e-10]
        print(f"\n  Pior caso (dc perde por {abs(worst_loss[3])} CNOTs):")
        for k, v in nz[:8]:
            print(f"    |{k:0{n_qubits}b}> = {v:.4f}")
        if len(nz) > 8:
            print(f"    ... ({len(nz)} amplitudes nao-nulas)")
    print()

    with open("resultados_benchmark.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["amostra", "tipo", "nnz",
                         "cx_ucge", "dep_ucge",
                         "cx_dc",   "dep_dc",
                         "delta_cx",
                         "fid_ucge", "fid_dc"])
        writer.writerows(csv_rows)
    print("  -> resultados_benchmark.csv salvo.")
    print()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Testes UCGE vs UCGE-DC")
    parser.add_argument("--no-benchmark", action="store_true",
                        help="Pula o benchmark estatistico")
    parser.add_argument("--samples", type=int, default=500,
                        help="Amostras no benchmark (default: 500)")
    parser.add_argument("--qubits", type=int, default=4,
                        help="Qubits no benchmark (default: 4)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Semente aleatoria (default: 42)")
    args = parser.parse_args()

    print()
    print("=" * 70)
    print("  UCGE ORIGINAL  vs  UCGE-DC")
    print("=" * 70)
    print()

    test_concrete()
    test_scaling()

    if not args.no_benchmark:
        test_benchmark(
            n_qubits=args.qubits,
            n_samples=args.samples,
            seed=args.seed,
        )
    else:
        print("(benchmark estatistico pulado)")
