#Lägg till --no-show för att spara grafen utan att öppna ett fönster.

import argparse
import os
from pathlib import Path
import time

folder = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(folder / ".matplotlib_cache"))

import matplotlib.pyplot as plt
import numpy as np

from thomas import thomas


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()

    # 1. FLOPRÄKNING
    # En flop är en flyttalsaddition, subtraktion, multiplikation eller
    # division. Vi räknar inte indexering, jämförelser eller kopiering.
    #
    # Framåtelimination, för var och en av N-1 rader:
    # multiplier = sub / diagonal             -> 1 division
    # diagonal -= multiplier * upper          -> 1 multiplikation + 1 subtraktion
    # rhs -= multiplier * previous_rhs        -> 1 multiplikation + 1 subtraktion
    # Totalt: 5(N-1) flops.
    #
    # Bakåtsubstitution:
    # Sista x beräknas med 1 division.
    # Övriga N-1 x kräver 1 multiplikation, 1 subtraktion och 1 division.
    # Totalt: 1 + 3(N-1) flops.
    #
    # Hela lösningen kostar alltså 8N-7 flops = O(N).
    # Indatakontroller, kopior och minnesallokering i vår implementation
    # kräver också O(N) arbete och ändrar därför inte tillväxtordningen.

    # 2. TIDSMÄTNING
    # Vi dubblerar N för att se hur lösningstiden växer.
    sizes = np.array([1000, 2000, 4000, 8000, 16000, 32000])
    rng = np.random.default_rng(42)
    timings = []
    repeat_counts = []

    for N in sizes:
        # Förbered testdata UTANFÖR tidtagningen, enligt samma upplägg
        # som verifieringen. En full N x N-matris skulle kosta O(N^2)
        # i lagring och störa undersökningen av vår O(N)-lösare.
        main_diag = N * np.ones(N) + rng.random(N)
        sub = rng.random(N - 1)
        upper = rng.random(N - 1)
        known_x = np.pi * np.ones(N)

        # Beräkna b = A @ known_x direkt från de tre diagonalerna.
        b = main_diag * known_x
        b[1:] += sub * known_x[:-1]
        b[:-1] += upper * known_x[1:]

        # Värm upp och uppskatta tiden för att välja antal upprepningar.
        start = time.process_time()
        x = thomas(sub, main_diag, upper, b)
        pilot_time = time.process_time() - start
        assert np.allclose(x, known_x, rtol=1e-12, atol=1e-12)

        # Varje mätomgång bör vara tillräckligt lång för att minska
        # påverkan från klockans upplösning. Sikta på minst 0.1 s.
        repetitions = min(200, max(3, int(np.ceil(0.1 / max(pilot_time, 1e-9)))))
        samples = []
        for _ in range(5):
            # process_time mäter processens CPU-tid, inte tiden den väntar.
            # Endast anropen till lösaren ligger i det uppmätta blocket.
            start = time.process_time()
            for _ in range(repetitions):
                thomas(sub, main_diag, upper, b)
            elapsed = time.process_time() - start
            samples.append(elapsed / repetitions)

        # Medianen av fem mätomgångar minskar inverkan av avvikande värden.
        timings.append(float(np.median(samples)))
        repeat_counts.append(repetitions)

    timings = np.array(timings)
    assert np.all(timings > 0)

    # Om t ungefär är C*N^p blir log(t) ungefär log(C) + p*log(N).
    # Lutningen i en log-log-graf uppskattar därför exponenten p.
    slope = np.polyfit(np.log(sizes), np.log(timings), 1)[0]
    lines = [
        "Milestone 1: kostnadsanalys av Thomas-algoritmen",
        "CPU-tid: median av fem omgångar, med upprepade lösningar per omgång.",
        "",
        f"{'N':>8} {'Flops (8N-7)':>16} {'CPU-tid/lösning (s)':>22} {'Tidskvot':>12} {'Upprepningar':>14}",
    ]
    for i, N in enumerate(sizes):
        ratio = '--' if i == 0 else f'{timings[i] / timings[i-1]:.4f}'
        lines.append(
            f'{N:8d} {8*N-7:16d} {timings[i]:22.4e} {ratio:>12} {repeat_counts[i]:14d}'
        )

    lines.extend([
        "",
        "Flopräkning:",
        f"Uppmätt log-log-lutning: {slope:.4f}. Förväntat värde för O(N): 1.",
    ])

    # 3. RITA MÄTDATA OCH EN REFERENSLINJE, SOM I LABB 2.1.
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.loglog(sizes, timings, 'o-', label='Thomas: measured CPU time')
    reference = timings[-1] * sizes / sizes[-1]
    ax.loglog(sizes, reference, 'k--', label='O(N) reference')
    ax.set_xlabel('System size N')
    ax.set_ylabel('CPU time per solve (s)')
    ax.set_title(f'Thomas algorithm: measured slope = {slope:.2f}')
    ax.grid(True, which='both', alpha=0.3)
    ax.legend()
    fig.tight_layout()

    # Spara båda resultaten bredvid skriptet.
    output = folder / 'resultat_cost_analysis_thomas.txt'
    figure_path = folder / 'figur_cost_analysis_thomas.png'
    result = '\n'.join(lines) + '\n'
    output.write_text(result, encoding='utf-8')
    fig.savefig(figure_path, dpi=180)
    print(result)
    print(f'Tabell sparad i: {output}')
    print(f'Graf sparad i: {figure_path}')
    if args.no_show:
        plt.close(fig)
    else:
        plt.show()


if __name__ == '__main__':
    main()

