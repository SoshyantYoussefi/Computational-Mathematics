#Thomas-algoritmen för ett tridiagonalt linjärt system Ax = b.
import numpy as np


def thomas(sub, main, upper, b):
    """Lös Ax = b utan att bygga en fullständig matris.

    Parametrar
    ----------
    sub : vektor med N-1 element
        Diagonalen under huvuddiagonalen: sub[i] = A[i+1, i].
    main : vektor med N element
        Huvuddiagonalen: main[i] = A[i, i].
    upper : vektor med N-1 element
        Diagonalen över huvuddiagonalen: upper[i] = A[i, i+1].
    b : vektor med N element
        Systemets högerled.

    Returnerar
    ----------
    x : NumPy-vektor med N element
        Den beräknade lösningen.

    Metoden är en specialiserad LU-metod utan pivotering. Den fungerar
    för projektets strikt diagonaldominanta testmatriser, men inte för
    alla inverterbara tridiagonala matriser. En nollpivot ger ett fel.
    Beräkningskostnad och lagringsbehov är båda O(N).
    """
    # Kopiera indata och använd flyttal. Då ändras inte användarens
    # ursprungliga vektorer när eliminationen uppdaterar värdena.
    lower = np.array(sub, dtype=float, copy=True)
    diagonal = np.array(main, dtype=float, copy=True)
    upper = np.array(upper, dtype=float, copy=True)
    rhs = np.array(b, dtype=float, copy=True)

    # Alla indata ska vara endimensionella vektorer med rätt längder.
    vectors = (lower, diagonal, upper, rhs)
    if any(vector.ndim != 1 for vector in vectors):
        raise ValueError("Alla indata måste vara endimensionella vektorer.")

    n = len(diagonal)
    if n == 0:
        raise ValueError("Systemet måste innehålla minst en ekvation.")
    if len(lower) != n - 1 or len(upper) != n - 1 or len(rhs) != n:
        raise ValueError("main och b ska ha N element; sub och upper N-1.")
    if any(not np.all(np.isfinite(vector)) for vector in vectors):
        raise ValueError("Indata får inte innehålla NaN eller oändliga värden.")

    # STEG 1: Framåtelimination.
    # Rad i-1 används för att eliminera elementet under diagonalen
    # på rad i. Eftersom matrisen är tridiagonal behöver bara ett
    # diagonalelement och ett högerledselement uppdateras per rad.
    for i in range(1, n):
        if diagonal[i - 1] == 0:
            raise np.linalg.LinAlgError(
                f"Nollpivot på rad {i - 1}; Thomas utan pivotering kan inte fortsätta."
            )

        # Multiplikatorn anger hur mycket av föregående rad som dras bort.
        multiplier = lower[i - 1] / diagonal[i - 1]
        diagonal[i] -= multiplier * upper[i - 1]
        rhs[i] -= multiplier * rhs[i - 1]

    if diagonal[-1] == 0:
        raise np.linalg.LinAlgError("Nollpivot på sista raden.")

    # STEG 2: Bakåtsubstitution.
    # Efter eliminationen är systemet övre bidiagonalt. Lös sista
    # ekvationen först och gå sedan bakåt, en rad i taget.
    x = np.zeros(n)
    x[-1] = rhs[-1] / diagonal[-1]
    for i in range(n - 2, -1, -1):
        x[i] = (rhs[i] - upper[i] * x[i + 1]) / diagonal[i]

    return x


if __name__ == "__main__":
    # Exempel: en tridiagonal 3x3-matris med 4 på huvuddiagonalen
    # och -1 på de två andra diagonalerna. Lösningen är [1, 1, 1].
    sub = np.array([-1.0, -1.0])
    main = np.array([4.0, 4.0, 4.0])
    upper = np.array([-1.0, -1.0])
    b = np.array([3.0, 2.0, 3.0])

    x = thomas(sub, main, upper, b)
    print("Beräknad lösning:", x)
    print("Förväntad lösning: [1. 1. 1.]")
