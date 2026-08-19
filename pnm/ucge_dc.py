"""
UCGE with optimal don't-care filling (UCGEDCInitialize).

Para cada slot dc (par de amplitudes com norma zero), o operador no
multiplexador pode ser qualquer SU(2) sem afetar o estado preparado.
Escolhemos os pares de amplitudes que minimizam o número de operadores
após _repetition_search, combinando duas estratégias:

1. Period fill: para cada período d=1,2,4,..., se os pares ativos são
   globalmente periódicos com período d, preenche os slots dc com o par
   ativo de mesmo resíduo (mod d).

2. Sibling fill: preenche cada slot dc com o par do irmão (k^1).

A estratégia escolhida é a que minimiza ops_after. Como o par original
(norma zero, operador = identidade) está sempre disponível implicitamente,
o resultado NUNCA é pior que o UCGE original.

Implementação: toda a lógica opera sobre o vetor de amplitudes children
diretamente. Não há conversão children -> mux -> children, eliminando o
bug de fase que afetaria amplitudes complexas.

Regra de propagação
-------------------
_disentangle_qubit recebe children_filled e parent_filled.
_apply_diagonal     recebe parent ORIGINAL.

Otimizações de desempenho
-------------------------
- _build_mux_matrices usa operações vetorizadas (numpy) em vez de loop por par
- _optimal_fill_children reutiliza mux_orig: só reconstrói as entradas que
  mudaram (os slots dc), em vez de recalcular o multiplexador inteiro a cada
  estratégia
- _update_mux_slots: reconstrução parcial O(|dc_slots|) em vez de O(m)
"""

import numpy as np
from qclib.state_preparation.ucge import UCGEInitialize


# ── Utilitários ───────────────────────────────────────────────────────────

def _build_mux_matrices(children):
    """
    Constrói lista de matrizes 2x2 do multiplexador a partir de children.
    Vetorizado: evita loop Python por par.
    """
    ch = np.asarray(children)
    n  = len(ch) // 2
    a0 = ch[0::2]          # amplitudes pares
    a1 = ch[1::2]          # amplitudes ímpares
    norms = np.sqrt(np.abs(a0)**2 + np.abs(a1)**2)

    mux = []
    for k in range(n):
        p = norms[k]
        if p < 1e-12:
            mux.append(np.eye(2))
        else:
            u0, u1 = a0[k] / p, a1[k] / p
            op = np.array([[u0, -np.conj(u1)],
                           [u1,  np.conj(u0)]])
            mux.append(np.conj(op).T)
    return mux


def _update_mux_slots(mux_base, children, slots):
    """
    Reconstrói apenas as entradas `slots` do multiplexador,
    copiando o restante de mux_base. O(|slots|) em vez de O(m).
    """
    mux = list(mux_base)          # cópia rasa — entradas são arrays imutáveis
    ch  = np.asarray(children)
    for k in slots:
        a0, a1 = ch[2*k], ch[2*k+1]
        p = np.sqrt(abs(a0)**2 + abs(a1)**2)
        if p < 1e-12:
            mux[k] = np.eye(2)
        else:
            u0, u1 = a0 / p, a1 / p
            op = np.array([[u0, -np.conj(u1)],
                           [u1,  np.conj(u0)]])
            mux[k] = np.conj(op).T
    return mux


def _ops_after(mux):
    """
    Simula _repetition_search + _find_operators_to_remove.
    Retorna número de operadores restantes após simplificação.
    """
    m = len(mux)
    if m <= 1:
        return m
    deleted = set()
    for d in [2**j for j in range(int(np.log2(m)))]:
        if not np.allclose(mux[d], mux[0]):
            continue
        candidate, idx, ok = set(), 0, True
        for _ in range(m // (2 * d)):
            if np.allclose(mux[idx:idx+d], mux[idx+d:idx+2*d]):
                candidate.update(range(idx+d, idx+2*d))
                idx += 2 * d
            else:
                ok = False
                break
        if ok:
            deleted.update(candidate)
    return m - len(deleted)


def _optimal_fill_children(children, dc_slots):
    """
    Retorna (children_filled, ops_after) com o preenchimento ótimo dos
    slots dc, operando diretamente sobre o vetor de amplitudes.

    Nunca pior que o original (identidade nos slots dc).

    Otimização: reutiliza mux_orig e reconstrói apenas os slots dc
    modificados (_update_mux_slots) em vez de reconstruir o mux inteiro.
    """
    dc_set = set(dc_slots)
    m      = len(children) // 2

    # baseline: mux original calculado uma única vez
    mux_orig = _build_mux_matrices(children)
    best_ops = _ops_after(mux_orig)
    best_ch  = np.array(children)

    # ── estratégia 1: period fill ─────────────────────────────────────
    for d in [2**j for j in range(int(np.log2(m)))]:
        template_ch = [None] * d
        template_op = [None] * d
        conflict = False

        for k in range(m):
            if k in dc_set:
                continue
            r = k % d
            if template_op[r] is None:
                template_op[r] = mux_orig[k]
                template_ch[r] = (children[2*k], children[2*k+1])
            elif not np.allclose(template_op[r], mux_orig[k]):
                conflict = True
                break
        if conflict:
            continue

        # aplica substituição apenas nos slots dc
        candidate = np.array(children)
        changed = []
        for k in dc_slots:
            r = k % d
            if template_ch[r] is not None:
                candidate[2*k]   = template_ch[r][0]
                candidate[2*k+1] = template_ch[r][1]
                changed.append(k)

        # reconstrói apenas as entradas que mudaram
        mux_cand = _update_mux_slots(mux_orig, candidate, changed)
        o = _ops_after(mux_cand)
        if o < best_ops:
            best_ops = o
            best_ch  = candidate

    # ── estratégia 2: sibling fill ────────────────────────────────────
    candidate = np.array(children)
    changed = []
    for k in dc_slots:
        sib = k ^ 1
        if sib not in dc_set:
            candidate[2*k]   = children[2*sib]
            candidate[2*k+1] = children[2*sib+1]
            changed.append(k)

    mux_cand = _update_mux_slots(mux_orig, candidate, changed)
    o = _ops_after(mux_cand)
    if o < best_ops:
        best_ops = o
        best_ch  = candidate

    return best_ch, best_ops


# ── Classe principal ──────────────────────────────────────────────────────

class UCGEDCInitialize(UCGEInitialize):
    """
    UCGE com preenchimento ótimo de don't-cares.

    Uso
    ---
        from ucge_dc import UCGEDCInitialize

        gate = UCGEDCInitialize(state_vector)
        circuit.append(gate.definition, circuit.qubits)

        UCGEDCInitialize.initialize(circuit, state_vector)
    """

    def __init__(self, params, label=None, opt_params=None):
        super().__init__(params, label=label, opt_params=opt_params)

    def _define_initialize(self):
        children   = self.params
        parent     = self._update_parent(children)
        tree_level = self.num_qubits
        r_gate     = self.target_state // 2

        while tree_level > 0:

            dc_slots = [k for k in range(len(children) // 2)
                        if np.linalg.norm([children[2*k], children[2*k+1]])
                        < 1e-12]

            if dc_slots:
                children_filled, _ = _optimal_fill_children(children, dc_slots)
                parent_filled      = self._update_parent(children_filled)

                bit_target, ucg = self._disentangle_qubit(
                    children_filled, parent_filled, r_gate, tree_level)
                # parent ORIGINAL: zeros dc propagam corretamente
                children = self._apply_diagonal(bit_target, parent, ucg)
            else:
                bit_target, ucg = self._disentangle_qubit(
                    children, parent, r_gate, tree_level)
                children = self._apply_diagonal(bit_target, parent, ucg)

            parent     = self._update_parent(children)
            r_gate     = r_gate // 2
            tree_level -= 1

        return self.circuit.inverse()

    @staticmethod
    def initialize(q_circuit, state, qubits=None, opt_params=None):
        gate = UCGEDCInitialize(state, opt_params=opt_params)
        if qubits is None:
            q_circuit.append(gate.definition, q_circuit.qubits)
        else:
            q_circuit.append(gate.definition, qubits)
