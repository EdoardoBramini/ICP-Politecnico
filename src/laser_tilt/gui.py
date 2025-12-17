from __future__ import annotations
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter.scrolledtext import ScrolledText

import numpy as np
from .icp_core import compute_from_files


class ICPGui(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Section ICP (measured → target) + Correction (T⁻¹)")
        self.geometry("920x700")

        self.target_path = tk.StringVar(value="")
        self.measured_path = tk.StringVar(value="")
        self.downsample = tk.StringVar(value="1")
        self.max_iter = tk.StringVar(value="50")
        self.tol = tk.StringVar(value="1e-8")

        self._last = None  # (res, Tcorr)
        self._build()

    def _build(self):
        frm = tk.Frame(self)
        frm.pack(fill="x", padx=10, pady=10)

        tk.Label(frm, text="Target (reference) file:").grid(row=0, column=0, sticky="w")
        tk.Entry(frm, textvariable=self.target_path, width=70).grid(row=0, column=1, padx=6)
        tk.Button(frm, text="Browse…", command=self._browse_target).grid(row=0, column=2)

        tk.Label(frm, text="Measured (current) file:").grid(row=1, column=0, sticky="w", pady=(8, 0))
        tk.Entry(frm, textvariable=self.measured_path, width=70).grid(row=1, column=1, padx=6, pady=(8, 0))
        tk.Button(frm, text="Browse…", command=self._browse_measured).grid(row=1, column=2, pady=(8, 0))

        opts = tk.Frame(self)
        opts.pack(fill="x", padx=10)

        tk.Label(opts, text="Downsample step:").grid(row=0, column=0, sticky="w")
        tk.Entry(opts, textvariable=self.downsample, width=8).grid(row=0, column=1, sticky="w", padx=6)

        tk.Label(opts, text="Max iterations:").grid(row=0, column=2, sticky="w", padx=(20, 0))
        tk.Entry(opts, textvariable=self.max_iter, width=8).grid(row=0, column=3, sticky="w", padx=6)

        tk.Label(opts, text="Tolerance:").grid(row=0, column=4, sticky="w", padx=(20, 0))
        tk.Entry(opts, textvariable=self.tol, width=12).grid(row=0, column=5, sticky="w", padx=6)

        tk.Button(opts, text="Compute ICP", command=self._compute).grid(row=0, column=6, padx=(20, 0))

        self.out = ScrolledText(self, height=30)
        self.out.pack(fill="both", expand=True, padx=10, pady=10)
        self._write("Seleziona Target e Measured, poi Compute ICP.\n")

        bottom = tk.Frame(self)
        bottom.pack(fill="x", padx=10, pady=(0, 10))
        tk.Button(bottom, text="Plot overlay", command=self._plot).pack(side="left")
        tk.Button(bottom, text="Copy report", command=self._copy).pack(side="left", padx=8)

    def _write(self, s: str):
        self.out.configure(state="normal")
        self.out.insert("end", s)
        self.out.configure(state="disabled")

    def _set_text(self, s: str):
        self.out.configure(state="normal")
        self.out.delete("1.0", "end")
        self.out.insert("end", s)
        self.out.configure(state="disabled")

    def _browse_target(self):
        p = filedialog.askopenfilename(title="Select target/reference file", filetypes=[("Text", "*.txt"), ("All", "*.*")])
        if p:
            self.target_path.set(p)

    def _browse_measured(self):
        p = filedialog.askopenfilename(title="Select measured/current file", filetypes=[("Text", "*.txt"), ("All", "*.*")])
        if p:
            self.measured_path.set(p)

    def _compute(self):
        tpath = self.target_path.get().strip()
        mpath = self.measured_path.get().strip()
        if not tpath or not mpath:
            messagebox.showwarning("Missing files", "Seleziona sia Target che Measured.")
            return

        try:
            ds = int(self.downsample.get().strip() or "1")
            mi = int(self.max_iter.get().strip() or "50")
            tol = float(self.tol.get().strip() or "1e-8")
        except ValueError:
            messagebox.showerror("Bad params", "Downsample/max_iter/tolerance non validi.")
            return

        try:
            res, Tcorr = compute_from_files(
                tpath, mpath,
                downsample_step=ds,
                max_iterations=mi,
                tolerance=tol,
            )
            self._last = (res, Tcorr)

            theta_corr = float(np.degrees(np.arctan2(Tcorr[1, 0], Tcorr[0, 0])))
            tx_corr, ty_corr = float(Tcorr[0, 2]), float(Tcorr[1, 2])

            report = (
                "=== ICP (measured → target) ===\n"
                f"theta_deg: {res.theta_deg:.6f}\n"
                f"t: [{res.tx:.6f}, {res.ty:.6f}]\n"
                f"rmse: {res.rmse:.6f}\n"
                f"iterations: {res.iterations}\n"
                f"T:\n{res.T}\n\n"
                "=== CORRECTION (apply T^-1) ===\n"
                f"theta_corr_deg: {theta_corr:.6f}\n"
                f"t_corr: [{tx_corr:.6f}, {ty_corr:.6f}]\n"
                f"T_corr:\n{Tcorr}\n"
            )
            self._set_text(report)
        except Exception as e:
            messagebox.showerror("ICP failed", str(e))

    def _plot(self):
        if not self._last:
            messagebox.showinfo("Nothing to plot", "Calcola ICP prima.")
            return

        import matplotlib.pyplot as plt
        from .io_points import load_points_txt, downsample_uniform

        res, _ = self._last
        target = downsample_uniform(load_points_txt(self.target_path.get().strip()), int(self.downsample.get()))
        meas = downsample_uniform(load_points_txt(self.measured_path.get().strip()), int(self.downsample.get()))

        # Applica T (measured -> target)
        R = res.T[:2, :2]
        t = res.T[:2, 2]
        corr = (meas @ R.T) + t

        plt.figure()
        plt.scatter(target[:,0], target[:,1], s=4, label="target")
        plt.scatter(meas[:,0], meas[:,1], s=4, label="measured")
        plt.scatter(corr[:,0], corr[:,1], s=4, label="measured aligned")
        plt.axis("equal")
        plt.title("ICP overlay: measured → target")
        plt.legend()
        plt.show()

    def _copy(self):
        txt = self.out.get("1.0", "end").strip()
        if not txt:
            return
        self.clipboard_clear()
        self.clipboard_append(txt)
        messagebox.showinfo("Copied", "Report copiato negli appunti.")


def main():
    app = ICPGui()
    app.mainloop()
