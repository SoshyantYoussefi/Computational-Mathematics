from pathlib import Path
import numpy as np
from thomas import thomas


def main():
    # 1. Välj storlekar på systemen som ska testas.
    # N är både antalet ekvationer och antalet obekanta.
    sizes = [1, 2, 10, 20, 50, 100, 500, 1000]

    # Ett fast frö gör att vi får samma slumpvärden vid varje körning.
    # Det gör resultaten reproducerbara och enklare att felsöka.
    rng = np.random.default_rng(42)

    lines = [
        "Milestone 1: verifiering av Thomas-algoritmen",
        "Känd lösning: x = pi * ones(N). Slumpfrö: 42.",
        "",
        f"{'N':>6} {'Lösningsfel (inf)':>20} {'Residual (inf)':>20} {'Relativ residual':>20}",
    ]

    for N in sizes:
        # 2. Skapa de tre diagonalerna enligt projektets testupplägg.
        # Slumptalen ligger mellan 0 och 1. Huvuddiagonalen är stor
        # jämfört med övriga element på samma rad. Matrisen är därför
        # strikt diagonaldominant och Thomas kan köras utan pivotering.
        main_diag = N * np.ones(N) + rng.random(N)
        sub = rng.random(N - 1)
        upper = rng.random(N - 1)

        # np.diag(v, k) placerar v på diagonal k:
        # k=-1: under huvuddiagonalen, k=0: huvuddiagonalen,
        # k=1: över huvuddiagonalen.
        # Den fullständiga matrisen används bara för verifieringen.
        # Thomas-lösaren själv arbetar bara med diagonalvektorerna.
        A = np.diag(sub, k=-1) + np.diag(main_diag) + np.diag(upper, k=1)

        # Kontrollera diagonaldominansen: |A_ii| ska vara större än
        # summan av absolutvärdena av övriga element på samma rad.
        off_diag_sum = np.zeros(N)
        off_diag_sum[1:] += np.abs(sub)
        off_diag_sum[:-1] += np.abs(upper)
        assert np.all(np.abs(main_diag) > off_diag_sum)

        # 3. Tillverka ett system med en känd lösning.
        # Eftersom b = A @ known_x vet vi i förväg att known_x löser Ax=b.
        known_x = np.pi * np.ones(N)
        b = A @ known_x

        # 4. Beräkna lösningen med vår egen Thomas-algoritm.
        x = thomas(sub, main_diag, upper, b)

        # 5. Mät två olika saker:
        # Lösningsfel: hur långt x ligger från den kända lösningen.
        # Residual: hur väl den beräknade lösningen uppfyller Ax=b.
        # Oändlighetsnormen för en vektor är dess största absolutvärde.
        solution_error = np.linalg.norm(x - known_x, ord=np.inf)
        residual = np.linalg.norm(A @ x - b, ord=np.inf)

        # Relativ residual gör residualen jämförbar mellan olika N.
        # En stor matris ger större b, så absolutresidualen kan öka
        # även om den relativa noggrannheten fortfarande är mycket god.
        relative_residual = residual / np.linalg.norm(b, ord=np.inf)

        # Kontrollera att lösningen är ändlig och nära det kända svaret.
        # Vi förväntar oss små avrundningsfel, inte exakt noll.
        assert np.all(np.isfinite(x)), f"Icke-ändlig lösning för N={N}"
        assert np.allclose(x, known_x, rtol=1e-12, atol=1e-12), (
            f"Verifieringen misslyckades för N={N}"
        )
        assert relative_residual < 1e-12, f"För stor residual för N={N}"

        # Vetenskaplig notation med fyra decimaler, som i labb 2.
        lines.append(
            f"{N:6d} {solution_error:20.4e} {residual:20.4e} {relative_residual:20.4e}"
        )

    lines.extend([
        "",
        "Alla tester godkända."
    ])

    # 6. Spara tabellen i projektmappen, oavsett arbetsmapp vid körning.
    result = "\n".join(lines) + "\n"
    output_path = Path(__file__).resolve().with_name("resultat_verify_thomas.txt")
    output_path.write_text(result, encoding="utf-8")
    print(result)
    print(f"Resultaten sparades i: {output_path}")


# Kör testerna när filen startas direkt, men inte när den importeras.
if __name__ == "__main__":
    main()
