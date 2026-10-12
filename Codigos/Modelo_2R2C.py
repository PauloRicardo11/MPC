"""
============================================================
Módulo: Modelo_2R2C.py

Descrição:
    Funções e estruturas destinadas à representação,
    discretização e simulação do modelo térmico 2R2C
    utilizado na identificação dinâmica da zona térmica
    em estudo.

    O módulo permite:

    - definir os parâmetros físicos do modelo 2R2C;
    - construir a representação em espaço de estados contínuo;
    - validar as matrizes contínuas;
    - discretizar o modelo através de retenção de ordem zero
      (ZOH);
    - validar as matrizes discretas resultantes;
    - simular a evolução temporal do modelo discreto;
    - validar estados iniciais e sequências de entradas;
    - apresentar resultados de forma organizada.

Modelo:

    Estados:
        x = [T_int, T_m]^T

    Entradas:
        u = [T_ext, PSum]^T

    Saída:
        y = T_int

    Equações contínuas:

        C_int * dT_int/dt =
            (T_ext - T_int) / R_ext
            +
            (T_m - T_int) / R_m

        C_m * dT_m/dt =
            K_HVAC * PSum
            -
            (T_m - T_int) / R_m

Autor:
    Paulo Ribeiro

Projeto:
    Plataforma de Processamento de Dados Experimentais
    Dissertação de Mestrado
============================================================
"""

from dataclasses import dataclass

import numpy as np
from scipy.signal import cont2discrete


# ============================================================
# Estruturas de dados
# ============================================================


@dataclass(frozen=True)
class Parametros2R2C:
    """
    Parâmetros físicos do modelo térmico 2R2C.

    Parâmetros
    ----------
    R_ext : float
        Resistência térmica equivalente entre o ambiente
        exterior e o ar interior.

    R_m : float
        Resistência térmica equivalente entre o ar interior
        e a massa térmica.

    C_int : float
        Capacitância térmica equivalente associada ao ar
        interior e aos elementos de resposta térmica rápida.

    C_m : float
        Capacitância térmica equivalente da massa térmica.

    K_HVAC : float
        Ganho térmico efetivo entre a potência elétrica
        medida do sistema AVAC e a potência térmica
        equivalente introduzida no modelo.
    """

    R_ext: float
    R_m: float
    C_int: float
    C_m: float
    K_HVAC: float


@dataclass(frozen=True)
class MatrizesEstadoContinuo:
    """
    Matrizes da representação contínua em espaço de estados.

    Representação:

        dx/dt = A x + B u
        y     = C x + D u
    """

    A: np.ndarray
    B: np.ndarray
    C: np.ndarray
    D: np.ndarray


@dataclass(frozen=True)
class MatrizesEstadoDiscreto:
    """
    Matrizes da representação discreta em espaço de estados.

    Representação:

        x[k+1] = Ad x[k] + Bd u[k]
        y[k]   = Cd x[k] + Dd u[k]

    Atributos
    ---------
    Ad : np.ndarray
        Matriz de transição dos estados.

    Bd : np.ndarray
        Matriz das entradas.

    Cd : np.ndarray
        Matriz de saída.

    Dd : np.ndarray
        Matriz de transmissão direta.

    periodo_amostragem_s : float
        Período de amostragem utilizado na discretização,
        expresso em segundos.
    """

    Ad: np.ndarray
    Bd: np.ndarray
    Cd: np.ndarray
    Dd: np.ndarray
    periodo_amostragem_s: float


@dataclass(frozen=True)
class ResultadoSimulacaoDiscreta:
    """
    Resultado da simulação temporal do modelo discreto 2R2C.

    Para N amostras de entrada:

        entradas.shape = (N, 2)
        estados.shape = (N + 1, 2)
        saidas.shape = (N, 1)

    O primeiro elemento de estados corresponde ao estado
    inicial x[0].

    Atributos
    ---------
    estados : np.ndarray
        Evolução temporal dos estados T_int e T_m.

    saidas : np.ndarray
        Evolução temporal da saída T_int.

    entradas : np.ndarray
        Sequência de entradas utilizada na simulação.

    periodo_amostragem_s : float
        Período de amostragem do modelo discreto.
    """

    estados: np.ndarray
    saidas: np.ndarray
    entradas: np.ndarray
    periodo_amostragem_s: float


# ============================================================
# Validação dos parâmetros físicos
# ============================================================


def validar_parametros(parametros: Parametros2R2C) -> None:
    """
    Valida os parâmetros físicos do modelo 2R2C.

    Todos os parâmetros devem ser valores numéricos, finitos
    e estritamente positivos.

    Parameters
    ----------
    parametros : Parametros2R2C
        Parâmetros do modelo térmico.

    Raises
    ------
    TypeError
        Caso o argumento não seja uma instância de
        Parametros2R2C.

    ValueError
        Caso algum parâmetro não seja finito ou não seja
        estritamente positivo.
    """

    if not isinstance(parametros, Parametros2R2C):
        raise TypeError(
            "Os parâmetros devem ser fornecidos através de "
            "uma instância de Parametros2R2C."
        )

    parametros_dict = {
        "R_ext": parametros.R_ext,
        "R_m": parametros.R_m,
        "C_int": parametros.C_int,
        "C_m": parametros.C_m,
        "K_HVAC": parametros.K_HVAC,
    }

    for nome, valor in parametros_dict.items():

        if not isinstance(
            valor,
            (int, float, np.integer, np.floating),
        ):
            raise TypeError(
                f"O parâmetro '{nome}' deve ser numérico."
            )

        if not np.isfinite(valor):
            raise ValueError(
                f"O parâmetro '{nome}' deve ser finito."
            )

        if valor <= 0:
            raise ValueError(
                f"O parâmetro '{nome}' deve ser "
                "estritamente positivo."
            )


# ============================================================
# Construção do modelo contínuo
# ============================================================


def construir_matrizes_continuas(
    parametros: Parametros2R2C,
) -> MatrizesEstadoContinuo:
    """
    Constrói as matrizes contínuas do modelo térmico 2R2C.

    O vetor de estados é definido por:

        x = [T_int, T_m]^T

    O vetor de entradas é:

        u = [T_ext, PSum]^T

    A saída considerada é:

        y = T_int

    Parameters
    ----------
    parametros : Parametros2R2C
        Parâmetros físicos do modelo.

    Returns
    -------
    MatrizesEstadoContinuo
        Estrutura contendo as matrizes A, B, C e D.
    """

    validar_parametros(parametros)

    R_ext = float(parametros.R_ext)
    R_m = float(parametros.R_m)
    C_int = float(parametros.C_int)
    C_m = float(parametros.C_m)
    K_HVAC = float(parametros.K_HVAC)

    A = np.array(
        [
            [
                -(
                    1.0 / (R_ext * C_int)
                    + 1.0 / (R_m * C_int)
                ),
                1.0 / (R_m * C_int),
            ],
            [
                1.0 / (R_m * C_m),
                -1.0 / (R_m * C_m),
            ],
        ],
        dtype=float,
    )

    B = np.array(
        [
            [
                1.0 / (R_ext * C_int),
                0.0,
            ],
            [
                0.0,
                K_HVAC / C_m,
            ],
        ],
        dtype=float,
    )

    C = np.array(
        [
            [1.0, 0.0],
        ],
        dtype=float,
    )

    D = np.array(
        [
            [0.0, 0.0],
        ],
        dtype=float,
    )

    matrizes = MatrizesEstadoContinuo(
        A=A,
        B=B,
        C=C,
        D=D,
    )

    validar_matrizes_continuas(matrizes)

    return matrizes


# ============================================================
# Validação das matrizes contínuas
# ============================================================


def validar_matrizes_continuas(
    matrizes: MatrizesEstadoContinuo,
) -> None:
    """
    Valida as matrizes contínuas do modelo 2R2C.

    São verificadas:

    - existência das quatro matrizes;
    - dimensões esperadas;
    - presença apenas de valores finitos.

    Dimensões esperadas:

        A : (2, 2)
        B : (2, 2)
        C : (1, 2)
        D : (1, 2)

    Parameters
    ----------
    matrizes : MatrizesEstadoContinuo
        Matrizes contínuas do modelo.

    Raises
    ------
    TypeError
        Caso o argumento não seja uma instância de
        MatrizesEstadoContinuo.

    ValueError
        Caso alguma matriz apresente dimensões incorretas
        ou valores não finitos.
    """

    if not isinstance(matrizes, MatrizesEstadoContinuo):
        raise TypeError(
            "As matrizes devem ser fornecidas através de "
            "uma instância de MatrizesEstadoContinuo."
        )

    dimensoes_esperadas = {
        "A": (2, 2),
        "B": (2, 2),
        "C": (1, 2),
        "D": (1, 2),
    }

    matrizes_dict = {
        "A": matrizes.A,
        "B": matrizes.B,
        "C": matrizes.C,
        "D": matrizes.D,
    }

    for nome, matriz in matrizes_dict.items():

        if not isinstance(matriz, np.ndarray):
            raise TypeError(
                f"A matriz '{nome}' deve ser do tipo "
                "numpy.ndarray."
            )

        if matriz.shape != dimensoes_esperadas[nome]:
            raise ValueError(
                f"A matriz '{nome}' apresenta dimensão "
                f"{matriz.shape}, mas era esperada "
                f"{dimensoes_esperadas[nome]}."
            )

        if not np.all(np.isfinite(matriz)):
            raise ValueError(
                f"A matriz '{nome}' contém valores "
                "não finitos."
            )


# ============================================================
# Apresentação das matrizes contínuas
# ============================================================


def mostrar_matrizes_continuas(
    matrizes: MatrizesEstadoContinuo,
) -> None:
    """
    Apresenta as matrizes contínuas do modelo 2R2C.

    Parameters
    ----------
    matrizes : MatrizesEstadoContinuo
        Matrizes contínuas a apresentar.
    """

    validar_matrizes_continuas(matrizes)

    print("========== MODELO 2R2C CONTÍNUO ==========\n")

    print("Matriz A:")
    print(matrizes.A)

    print("\nMatriz B:")
    print(matrizes.B)

    print("\nMatriz C:")
    print(matrizes.C)

    print("\nMatriz D:")
    print(matrizes.D)

    print("\nDimensões:")
    print(f"A = {matrizes.A.shape}")
    print(f"B = {matrizes.B.shape}")
    print(f"C = {matrizes.C.shape}")
    print(f"D = {matrizes.D.shape}")

    print("\n===========================================")


# ============================================================
# Validação do período de amostragem
# ============================================================


def validar_periodo_amostragem(
    periodo_amostragem_s: float,
) -> float:
    """
    Valida o período de amostragem utilizado na discretização.

    O período de amostragem deve ser numérico, finito e
    estritamente positivo.

    Parameters
    ----------
    periodo_amostragem_s : float
        Período de amostragem em segundos.

    Returns
    -------
    float
        Período de amostragem validado.

    Raises
    ------
    TypeError
        Caso o período de amostragem não seja numérico.

    ValueError
        Caso o período de amostragem não seja finito ou
        seja inferior ou igual a zero.
    """

    if not isinstance(
        periodo_amostragem_s,
        (int, float, np.integer, np.floating),
    ):
        raise TypeError(
            "O período de amostragem deve ser um valor numérico."
        )

    periodo_amostragem_s = float(periodo_amostragem_s)

    if not np.isfinite(periodo_amostragem_s):
        raise ValueError(
            "O período de amostragem deve ser finito."
        )

    if periodo_amostragem_s <= 0:
        raise ValueError(
            "O período de amostragem deve ser "
            "estritamente positivo."
        )

    return periodo_amostragem_s


# ============================================================
# Discretização do modelo
# ============================================================


def discretizar_modelo(
    matrizes_continuas: MatrizesEstadoContinuo,
    periodo_amostragem_s: float,
) -> MatrizesEstadoDiscreto:
    """
    Discretiza o modelo contínuo 2R2C utilizando retenção
    de ordem zero (Zero-Order Hold - ZOH).

    A representação contínua:

        dx/dt = A x + B u
        y     = C x + D u

    é convertida para:

        x[k+1] = Ad x[k] + Bd u[k]
        y[k]   = Cd x[k] + Dd u[k]

    Parameters
    ----------
    matrizes_continuas : MatrizesEstadoContinuo
        Matrizes A, B, C e D do modelo contínuo.

    periodo_amostragem_s : float
        Período de amostragem em segundos.

    Returns
    -------
    MatrizesEstadoDiscreto
        Matrizes discretas Ad, Bd, Cd e Dd, juntamente com
        o período de amostragem utilizado.
    """

    validar_matrizes_continuas(matrizes_continuas)

    periodo_amostragem_s = validar_periodo_amostragem(
        periodo_amostragem_s
    )

    sistema_continuo = (
        matrizes_continuas.A,
        matrizes_continuas.B,
        matrizes_continuas.C,
        matrizes_continuas.D,
    )

    Ad, Bd, Cd, Dd, dt = cont2discrete(
        sistema_continuo,
        dt=periodo_amostragem_s,
        method="zoh",
    )

    matrizes_discretas = MatrizesEstadoDiscreto(
        Ad=np.asarray(Ad, dtype=float),
        Bd=np.asarray(Bd, dtype=float),
        Cd=np.asarray(Cd, dtype=float),
        Dd=np.asarray(Dd, dtype=float),
        periodo_amostragem_s=float(dt),
    )

    validar_matrizes_discretas(matrizes_discretas)

    return matrizes_discretas


# ============================================================
# Validação das matrizes discretas
# ============================================================


def validar_matrizes_discretas(
    matrizes: MatrizesEstadoDiscreto,
) -> None:
    """
    Valida a representação discreta do modelo 2R2C.

    São verificadas:

    - dimensões das matrizes;
    - presença apenas de valores finitos;
    - validade do período de amostragem.

    Dimensões esperadas:

        Ad : (2, 2)
        Bd : (2, 2)
        Cd : (1, 2)
        Dd : (1, 2)

    Parameters
    ----------
    matrizes : MatrizesEstadoDiscreto
        Matrizes discretas do modelo.
    """

    if not isinstance(matrizes, MatrizesEstadoDiscreto):
        raise TypeError(
            "As matrizes devem ser fornecidas através de "
            "uma instância de MatrizesEstadoDiscreto."
        )

    validar_periodo_amostragem(
        matrizes.periodo_amostragem_s
    )

    dimensoes_esperadas = {
        "Ad": (2, 2),
        "Bd": (2, 2),
        "Cd": (1, 2),
        "Dd": (1, 2),
    }

    matrizes_dict = {
        "Ad": matrizes.Ad,
        "Bd": matrizes.Bd,
        "Cd": matrizes.Cd,
        "Dd": matrizes.Dd,
    }

    for nome, matriz in matrizes_dict.items():

        if not isinstance(matriz, np.ndarray):
            raise TypeError(
                f"A matriz '{nome}' deve ser do tipo "
                "numpy.ndarray."
            )

        if matriz.shape != dimensoes_esperadas[nome]:
            raise ValueError(
                f"A matriz '{nome}' apresenta dimensão "
                f"{matriz.shape}, mas era esperada "
                f"{dimensoes_esperadas[nome]}."
            )

        if not np.all(np.isfinite(matriz)):
            raise ValueError(
                f"A matriz '{nome}' contém valores "
                "não finitos."
            )


# ============================================================
# Apresentação das matrizes discretas
# ============================================================


def mostrar_matrizes_discretas(
    matrizes: MatrizesEstadoDiscreto,
) -> None:
    """
    Apresenta as matrizes discretas do modelo 2R2C.

    Parameters
    ----------
    matrizes : MatrizesEstadoDiscreto
        Matrizes discretas a apresentar.
    """

    validar_matrizes_discretas(matrizes)

    print("========== MODELO 2R2C DISCRETO ==========\n")

    print(
        "Período de amostragem: "
        f"{matrizes.periodo_amostragem_s:.2f} s"
    )

    print("\nMatriz Ad:")
    print(matrizes.Ad)

    print("\nMatriz Bd:")
    print(matrizes.Bd)

    print("\nMatriz Cd:")
    print(matrizes.Cd)

    print("\nMatriz Dd:")
    print(matrizes.Dd)

    print("\nDimensões:")
    print(f"Ad = {matrizes.Ad.shape}")
    print(f"Bd = {matrizes.Bd.shape}")
    print(f"Cd = {matrizes.Cd.shape}")
    print(f"Dd = {matrizes.Dd.shape}")

    print("\n===========================================")


# ============================================================
# Validação do estado inicial
# ============================================================


def validar_estado_inicial(
    estado_inicial: np.ndarray,
) -> np.ndarray:
    """
    Valida o estado inicial utilizado na simulação.

    O vetor de estado deve apresentar a forma:

        x[0] = [T_int, T_m]

    Parameters
    ----------
    estado_inicial : np.ndarray
        Estado inicial do modelo.

    Returns
    -------
    np.ndarray
        Estado inicial convertido para um vetor do tipo float
        com dimensão (2,).

    Raises
    ------
    TypeError
        Caso o estado inicial não possa ser convertido para
        uma matriz NumPy.

    ValueError
        Caso o estado inicial não apresente dimensão (2,) ou
        contenha valores não finitos.
    """

    try:
        estado_inicial = np.asarray(
            estado_inicial,
            dtype=float,
        )
    except (TypeError, ValueError) as erro:
        raise TypeError(
            "O estado inicial deve conter valores numéricos."
        ) from erro

    if estado_inicial.shape != (2,):
        raise ValueError(
            "O estado inicial deve apresentar dimensão (2,), "
            "correspondente a [T_int, T_m]."
        )

    if not np.all(np.isfinite(estado_inicial)):
        raise ValueError(
            "O estado inicial contém valores não finitos."
        )

    return estado_inicial


# ============================================================
# Validação das entradas da simulação
# ============================================================


def validar_entradas_discretas(
    entradas: np.ndarray,
) -> np.ndarray:
    """
    Valida a sequência de entradas do modelo discreto.

    Cada linha deve apresentar:

        u[k] = [T_ext[k], PSum[k]]

    A matriz deve possuir dimensão:

        (N, 2)

    onde N representa o número de instantes simulados.

    Parameters
    ----------
    entradas : np.ndarray
        Sequência temporal das entradas.

    Returns
    -------
    np.ndarray
        Matriz de entradas validada e convertida para float.

    Raises
    ------
    TypeError
        Caso as entradas não sejam numéricas.

    ValueError
        Caso a matriz não possua dimensão (N, 2), esteja vazia
        ou contenha valores não finitos.
    """

    try:
        entradas = np.asarray(
            entradas,
            dtype=float,
        )
    except (TypeError, ValueError) as erro:
        raise TypeError(
            "As entradas devem conter valores numéricos."
        ) from erro

    if entradas.ndim != 2:
        raise ValueError(
            "As entradas devem ser fornecidas através de "
            "uma matriz bidimensional."
        )

    if entradas.shape[1] != 2:
        raise ValueError(
            "A matriz de entradas deve possuir exatamente "
            "duas colunas: [T_ext, PSum]."
        )

    if entradas.shape[0] == 0:
        raise ValueError(
            "A matriz de entradas não pode estar vazia."
        )

    if not np.all(np.isfinite(entradas)):
        raise ValueError(
            "A matriz de entradas contém valores não finitos."
        )

    return entradas


# ============================================================
# Simulação do modelo discreto
# ============================================================


def simular_modelo_discreto(
    matrizes: MatrizesEstadoDiscreto,
    entradas: np.ndarray,
    estado_inicial: np.ndarray,
) -> ResultadoSimulacaoDiscreta:
    """
    Simula a evolução temporal do modelo discreto 2R2C.

    O modelo é propagado através das equações:

        x[k+1] = Ad x[k] + Bd u[k]

        y[k] = Cd x[k] + Dd u[k]

    Para N amostras de entrada são calculados:

        N + 1 estados
        N saídas

    O primeiro estado corresponde ao estado inicial fornecido.

    Parameters
    ----------
    matrizes : MatrizesEstadoDiscreto
        Matrizes do modelo discreto.

    entradas : np.ndarray
        Matriz de entradas com dimensão (N, 2), sendo:

            coluna 0 -> T_ext
            coluna 1 -> PSum

    estado_inicial : np.ndarray
        Estado inicial:

            [T_int_inicial, T_m_inicial]

    Returns
    -------
    ResultadoSimulacaoDiscreta
        Estrutura contendo estados, saídas, entradas e período
        de amostragem.
    """

    validar_matrizes_discretas(matrizes)

    entradas = validar_entradas_discretas(
        entradas
    )

    estado_inicial = validar_estado_inicial(
        estado_inicial
    )

    numero_amostras = entradas.shape[0]

    estados = np.zeros(
        (numero_amostras + 1, 2),
        dtype=float,
    )

    saidas = np.zeros(
        (numero_amostras, 1),
        dtype=float,
    )

    estados[0] = estado_inicial

    for k in range(numero_amostras):

        saidas[k] = (
            matrizes.Cd @ estados[k]
            + matrizes.Dd @ entradas[k]
        )

        estados[k + 1] = (
            matrizes.Ad @ estados[k]
            + matrizes.Bd @ entradas[k]
        )

    if not np.all(np.isfinite(estados)):
        raise ValueError(
            "A simulação produziu estados não finitos."
        )

    if not np.all(np.isfinite(saidas)):
        raise ValueError(
            "A simulação produziu saídas não finitas."
        )

    return ResultadoSimulacaoDiscreta(
        estados=estados,
        saidas=saidas,
        entradas=entradas.copy(),
        periodo_amostragem_s=matrizes.periodo_amostragem_s,
    )


# ============================================================
# Validação do resultado da simulação
# ============================================================


def validar_resultado_simulacao(
    resultado: ResultadoSimulacaoDiscreta,
) -> None:
    """
    Valida estruturalmente o resultado de uma simulação.

    Parameters
    ----------
    resultado : ResultadoSimulacaoDiscreta
        Resultado da simulação a validar.
    """

    if not isinstance(
        resultado,
        ResultadoSimulacaoDiscreta,
    ):
        raise TypeError(
            "O resultado deve ser uma instância de "
            "ResultadoSimulacaoDiscreta."
        )

    entradas = validar_entradas_discretas(
        resultado.entradas
    )

    numero_amostras = entradas.shape[0]

    if resultado.estados.shape != (
        numero_amostras + 1,
        2,
    ):
        raise ValueError(
            "A matriz de estados apresenta dimensões "
            "incompatíveis com o número de entradas."
        )

    if resultado.saidas.shape != (
        numero_amostras,
        1,
    ):
        raise ValueError(
            "A matriz de saídas apresenta dimensões "
            "incompatíveis com o número de entradas."
        )

    if not np.all(np.isfinite(resultado.estados)):
        raise ValueError(
            "A matriz de estados contém valores não finitos."
        )

    if not np.all(np.isfinite(resultado.saidas)):
        raise ValueError(
            "A matriz de saídas contém valores não finitos."
        )

    validar_periodo_amostragem(
        resultado.periodo_amostragem_s
    )


# ============================================================
# Apresentação do resumo da simulação
# ============================================================


def mostrar_resumo_simulacao(
    resultado: ResultadoSimulacaoDiscreta,
) -> None:
    """
    Apresenta um resumo estrutural e numérico da simulação.

    Parameters
    ----------
    resultado : ResultadoSimulacaoDiscreta
        Resultado da simulação discreta.
    """

    validar_resultado_simulacao(resultado)

    numero_amostras = resultado.entradas.shape[0]

    print("========== SIMULAÇÃO DISCRETA 2R2C ==========\n")

    print(
        f"Número de entradas: {numero_amostras}"
    )

    print(
        "Período de amostragem: "
        f"{resultado.periodo_amostragem_s:.2f} s"
    )

    print("\nDimensões:")

    print(
        f"Entradas = {resultado.entradas.shape}"
    )

    print(
        f"Estados = {resultado.estados.shape}"
    )

    print(
        f"Saídas = {resultado.saidas.shape}"
    )

    print("\nEstado inicial:")
    print(resultado.estados[0])

    print("\nEstado final:")
    print(resultado.estados[-1])

    print("\nPrimeira saída:")
    print(resultado.saidas[0])

    print("\nÚltima saída:")
    print(resultado.saidas[-1])

    print("\n==============================================")