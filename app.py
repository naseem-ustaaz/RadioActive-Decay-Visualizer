import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from dataclasses import dataclass
from typing import Optional

# -------------------------
# Node + LinkedList classes
# -------------------------
@dataclass
class Node:
    atom_id: int
    time_step: int
    next: Optional['Node'] = None

class DecayLinkedList:
    def __init__(self):
        self.head = None
        self.tail = None

    def append(self, atom_id, time_step):
        node = Node(atom_id, time_step)
        if not self.head:
            self.head = node
            self.tail = node
        else:
            self.tail.next = node
            self.tail = node

    def to_list(self, limit=50):
        out = []
        cur = self.head
        count = 0
        while cur and count < limit:
            out.append((cur.atom_id, cur.time_step))
            cur = cur.next
            count += 1
        return out

# -------------------------
# Simulation logic
# -------------------------
def simulate_decay(n_atoms, half_life, n_steps, delta_t=1.0):
    p = 1 - 2 ** (-delta_t / half_life)
    alive = list(range(1, n_atoms + 1))
    linked = DecayLinkedList()

    remaining = []
    decayed_each = []

    for t in range(1, n_steps + 1):
        if len(alive) == 0:
            remaining.append(0)
            decayed_each.append(0)
            continue

        rand_vals = np.random.random(len(alive))
        mask = rand_vals < p

        decayed_atoms = [alive[i] for i in range(len(alive)) if mask[i]]
        for a in decayed_atoms:
            linked.append(a, t)

        alive = [alive[i] for i in range(len(alive)) if not mask[i]]

        remaining.append(len(alive))
        decayed_each.append(len(decayed_atoms))

    df = pd.DataFrame({
        "time_step": range(1, n_steps + 1),
        "remaining": remaining,
        "decayed": decayed_each,
    })

    df["cumulative_decayed"] = df["decayed"].cumsum()

    return df, linked, p

# -------------------------
# Streamlit UI
# -------------------------
st.title("🔬 Radioactive Decay Visualizer")

st.sidebar.header("Parameters")
n_atoms = st.sidebar.number_input("Initial atoms", 1, 2000, 200)
half_life = st.sidebar.number_input("Half life", 0.01, 100.0, 5.0)
n_steps = st.sidebar.number_input("Time steps", 1, 500, 50)
delta_t = st.sidebar.number_input("Δt", 0.001, 10.0, 1.0)

if st.sidebar.button("Run Simulation"):
    df, linked, p = simulate_decay(n_atoms, half_life, n_steps, delta_t)

    st.subheader(f"Decay Probability p = {p:.6f}")

    fig, ax = plt.subplots()
    ax.plot(df["time_step"], df["remaining"])
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Remaining Atoms")
    st.pyplot(fig)

    st.subheader("Decay Table")
    st.dataframe(df)

    st.subheader("LinkedList (First 50 decays)")
    st.write(pd.DataFrame(linked.to_list(), columns=["Atom ID", "Time Step"]))
