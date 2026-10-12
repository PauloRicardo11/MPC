"""
============================================================
Módulo: Modelo_2R2C_Coeficientes.py

Descrição:
    Representação alternativa do modelo térmico 2R2C através
    dos coeficientes dinâmicos diretamente presentes nas
    matrizes de espaço de estados.

    Esta parametrização é definida por:

        a1 = 1 / (R_ext * C_int)
        a2 = 1 / (R_m * C_int)
        a3 = 1 / (R_m * C_m)
        b_HVAC = K_HVAC / C_m

    permitindo representar o modelo contínuo segundo:

        A = [
            [-(a1 + a2),  a2],
            [a3,          -a3]
        ]

        B = [
            [a1,      0],
            [0,   b_HVAC]
        ]

    As entradas consideradas são:

        u = [T_ext, P_el]

    onde:

        T_ext : temperatura exterior [°C]
        P_el  : potência elétrica medida [W]

    Esta representação não altera a estrutura física 2R2C.
    Constitui apenas uma parametrização dinâmica equivalente,
    particularmente útil quando os parâmetros físicos
    individuais não são estruturalmente identificáveis de
    forma única a partir dos dados experimentais disponíveis.

Autor:
    Paulo Ribeiro

Projeto:
    Plataforma de Processamento de Dados Experimentais
    Dissertação de Mestrado
============================================================
"""

from dataclasses import dataclass

import numpy as np

from Codigos.Modelo_2R2C import (
    Parametros2R2C,
    MatrizesEstadoContinuo,
    validar_parametros,
    validar_matrizes_continuas,
)


# ============================================================
# Estruturas de dados
# ============================================================


@dataclass(frozen=True)
class Coeficientes2R2C:
    """
    Coeficientes dinâmicos do modelo térmico 2R2C.

    Atributos
    ---------
    a1 : float
        Coeficiente associado à troca térmica entre o
        ambiente exterior e o ar interior.

        a1 = 1 / (R_ext * C_int)

    a2 : float
        Coeficiente associado à troca térmica entre a massa
        térmica e o ar interior.

        a2 = 1 / (R_m * C_int)

    a3 : float
        Coeficiente associado à dinâmica da massa térmica.

        a3 = 1 / (R_m * C_m)

    b_HVAC : float
        Ganho dinâmico efetivo entre a potência elétrica
        medida e a evolução da temperatura da massa térmica.

        b_HVAC = K_HVAC / C_m
    """

    a1: float
    a2: float
    a3: float
    b_HVAC: float


# ============================================================
# Validação dos coeficientes
# ============================================================


def validar_coeficientes(
    coeficientes: Coeficientes2R2C,
) -> None:
    """
    Valida os coeficientes dinâmicos do modelo 2R2C.

    Todos os coeficientes devem ser:

    - numéricos;
    - finitos;
    - estritamente positivos.

    Parameters
    ----------
    coeficientes : Coeficientes2R2C
        Coeficientes a validar.

    Raises
    ------
    TypeError
        Caso seja fornecido um tipo incorreto ou algum
        coeficiente não seja numérico.

    ValueError
        Caso algum coeficiente não seja finito ou seja
        inferior ou igual a zero.
    """

    if not isinstance(
        coeficientes,
        Coeficientes2R2C,
    ):
        raise TypeError(
            "Os coeficientes devem ser fornecidos através "
            "de uma instância de Coeficientes2R2C."
        )

    valores = {
        "a1": coeficientes.a1,
        "a2": coeficientes.a2,
        "a3": coeficientes.a3,
        "b_HVAC": coeficientes.b_HVAC,
    }

    for nome, valor in valores.items():

        if not isinstance(
            valor,
            (int, float, np.integer, np.floating),
        ):
            raise TypeError(
                f"O coeficiente '{nome}' deve ser numérico."
            )

        valor_float = float(
            valor
        )

        if not np.isfinite(
            valor_float
        ):
            raise ValueError(
                f"O coeficiente '{nome}' deve ser finito."
            )

        if valor_float <= 0:
            raise ValueError(
                f"O coeficiente '{nome}' deve ser "
                "estritamente positivo."
            )


# ============================================================
# Conversão dos parâmetros físicos
# ============================================================


def converter_parametros_para_coeficientes(
    parametros: Parametros2R2C,
) -> Coeficientes2R2C:
    """
    Converte os parâmetros físicos do modelo 2R2C nos quatro
    coeficientes dinâmicos equivalentes.

    Parameters
    ----------
    parametros : Parametros2R2C
        Parâmetros físicos:

            R_ext
            R_m
            C_int
            C_m
            K_HVAC

    Returns
    -------
    Coeficientes2R2C
        Coeficientes dinâmicos equivalentes.
    """

    validar_parametros(
        parametros
    )

    a1 = (
        1.0
        / (
            float(parametros.R_ext)
            * float(parametros.C_int)
        )
    )

    a2 = (
        1.0
        / (
            float(parametros.R_m)
            * float(parametros.C_int)
        )
    )

    a3 = (
        1.0
        / (
            float(parametros.R_m)
            * float(parametros.C_m)
        )
    )

    b_hvac = (
        float(parametros.K_HVAC)
        / float(parametros.C_m)
    )

    coeficientes = Coeficientes2R2C(
        a1=a1,
        a2=a2,
        a3=a3,
        b_HVAC=b_hvac,
    )

    validar_coeficientes(
        coeficientes
    )

    return coeficientes


# ============================================================
# Construção das matrizes contínuas
# ============================================================


def construir_matrizes_continuas_coeficientes(
    coeficientes: Coeficientes2R2C,
) -> MatrizesEstadoContinuo:
    """
    Constrói as matrizes contínuas do modelo 2R2C diretamente
    a partir dos quatro coeficientes dinâmicos.

    O vetor de estados é:

        x = [T_int, T_m]

    O vetor de entradas é:

        u = [T_ext, P_el]

    A saída é:

        y = T_int

    Parameters
    ----------
    coeficientes : Coeficientes2R2C
        Coeficientes dinâmicos do modelo.

    Returns
    -------
    MatrizesEstadoContinuo
        Estrutura contendo as matrizes A, B, C e D.
    """

    validar_coeficientes(
        coeficientes
    )

    a1 = float(
        coeficientes.a1
    )

    a2 = float(
        coeficientes.a2
    )

    a3 = float(
        coeficientes.a3
    )

    b_hvac = float(
        coeficientes.b_HVAC
    )

    A = np.array(
        [
            [
                -(a1 + a2),
                a2,
            ],
            [
                a3,
                -a3,
            ],
        ],
        dtype=float,
    )

    B = np.array(
        [
            [
                a1,
                0.0,
            ],
            [
                0.0,
                b_hvac,
            ],
        ],
        dtype=float,
    )

    C = np.array(
        [
            [
                1.0,
                0.0,
            ],
        ],
        dtype=float,
    )

    D = np.array(
        [
            [
                0.0,
                0.0,
            ],
        ],
        dtype=float,
    )

    matrizes = MatrizesEstadoContinuo(
        A=A,
        B=B,
        C=C,
        D=D,
    )

    validar_matrizes_continuas(
        matrizes
    )

    return matrizes


# ============================================================
# Apresentação dos coeficientes
# ============================================================


def mostrar_coeficientes(
    coeficientes: Coeficientes2R2C,
) -> None:
    """
    Apresenta os coeficientes dinâmicos do modelo 2R2C.

    Parameters
    ----------
    coeficientes : Coeficientes2R2C
        Coeficientes a apresentar.
    """

    validar_coeficientes(
        coeficientes
    )

    print(
        "========== COEFICIENTES DINÂMICOS 2R2C ==========\n"
    )

    print(
        f"a1     = {coeficientes.a1:.12e} 1/s"
    )

    print(
        f"a2     = {coeficientes.a2:.12e} 1/s"
    )

    print(
        f"a3     = {coeficientes.a3:.12e} 1/s"
    )

    print(
        f"b_HVAC = "
        f"{coeficientes.b_HVAC:.12e} K/(W.s)"
    )

    print(
        "\n=================================================="
    )