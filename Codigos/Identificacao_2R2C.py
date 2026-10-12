"""
============================================================
Módulo: Identificacao_2R2C.py

Descrição:
    Funções e estruturas destinadas à preparação da
    identificação dos parâmetros do modelo térmico 2R2C.

    O módulo permite:

    - definir limites físicos para os parâmetros;
    - representar os parâmetros em escala logarítmica;
    - converter entre parâmetros físicos e log10;
    - validar vetores de parâmetros utilizados na identificação;
    - simular o modelo para um conjunto candidato de parâmetros;
    - calcular os resíduos entre temperatura medida e simulada.

    Nesta fase, o módulo não executa ainda qualquer algoritmo
    de otimização.

Parâmetros identificáveis:

    theta = [
        R_ext,
        R_m,
        C_int,
        C_m,
        K_HVAC
    ]

Convenção das unidades:

    R_ext, R_m : K/W
    C_int, C_m : J/K
    K_HVAC     : ganho térmico efetivo
    PSum       : W
    T_int      : °C
    T_ext      : °C

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
    construir_matrizes_continuas,
    discretizar_modelo,
    simular_modelo_discreto,
    validar_estado_inicial,
    validar_entradas_discretas,
    validar_periodo_amostragem,
)


# ============================================================
# Estruturas de dados
# ============================================================


@dataclass(frozen=True)
class LimitesIdentificacao2R2C:
    """
    Limites físicos dos parâmetros utilizados na identificação.

    Atributos
    ---------
    inferiores : np.ndarray
        Limites inferiores em unidades físicas.

    superiores : np.ndarray
        Limites superiores em unidades físicas.
    """

    inferiores: np.ndarray
    superiores: np.ndarray


# ============================================================
# Limites padrão da identificação
# ============================================================


def criar_limites_padrao() -> LimitesIdentificacao2R2C:
    """
    Cria os limites físicos inicialmente adotados para a
    identificação do modelo térmico 2R2C.

    Ordem dos parâmetros:

        [R_ext, R_m, C_int, C_m, K_HVAC]

    Returns
    -------
    LimitesIdentificacao2R2C
        Limites inferiores e superiores em unidades físicas.
    """

    inferiores = np.array(
        [
            1.0e-4,  # R_ext [K/W]
            1.0e-4,  # R_m [K/W]
            1.0e4,   # C_int [J/K]
            1.0e6,   # C_m [J/K]
            0.5,     # K_HVAC [-]
        ],
        dtype=float,
    )

    superiores = np.array(
        [
            1.0e-1,  # R_ext [K/W]
            1.0e-1,  # R_m [K/W]
            1.0e7,   # C_int [J/K]
            1.0e9,   # C_m [J/K]
            8.0,     # K_HVAC [-]
        ],
        dtype=float,
    )

    limites = LimitesIdentificacao2R2C(
        inferiores=inferiores,
        superiores=superiores,
    )

    validar_limites_identificacao(limites)

    return limites


# ============================================================
# Validação dos limites
# ============================================================


def validar_limites_identificacao(
    limites: LimitesIdentificacao2R2C,
) -> None:
    """
    Valida os limites utilizados na identificação.

    Parameters
    ----------
    limites : LimitesIdentificacao2R2C
        Limites físicos dos cinco parâmetros.

    Raises
    ------
    TypeError
        Caso os limites não sejam fornecidos através da
        estrutura esperada.

    ValueError
        Caso as dimensões sejam incorretas, existam valores
        não finitos, não positivos ou os limites superiores
        não sejam maiores do que os inferiores.
    """

    if not isinstance(
        limites,
        LimitesIdentificacao2R2C,
    ):
        raise TypeError(
            "Os limites devem ser fornecidos através de uma "
            "instância de LimitesIdentificacao2R2C."
        )

    if limites.inferiores.shape != (5,):
        raise ValueError(
            "Os limites inferiores devem apresentar "
            "dimensão (5,)."
        )

    if limites.superiores.shape != (5,):
        raise ValueError(
            "Os limites superiores devem apresentar "
            "dimensão (5,)."
        )

    if not np.all(np.isfinite(limites.inferiores)):
        raise ValueError(
            "Os limites inferiores contêm valores "
            "não finitos."
        )

    if not np.all(np.isfinite(limites.superiores)):
        raise ValueError(
            "Os limites superiores contêm valores "
            "não finitos."
        )

    if np.any(limites.inferiores <= 0):
        raise ValueError(
            "Todos os limites inferiores devem ser "
            "estritamente positivos."
        )

    if np.any(
        limites.superiores
        <= limites.inferiores
    ):
        raise ValueError(
            "Cada limite superior deve ser maior do que "
            "o respetivo limite inferior."
        )


# ============================================================
# Conversão de parâmetros físicos para log10
# ============================================================


def converter_parametros_para_log10(
    parametros: Parametros2R2C,
) -> np.ndarray:
    """
    Converte os parâmetros físicos do modelo para escala log10.

    Ordem utilizada:

        [R_ext, R_m, C_int, C_m, K_HVAC]

    Parameters
    ----------
    parametros : Parametros2R2C
        Parâmetros físicos do modelo.

    Returns
    -------
    np.ndarray
        Vetor de parâmetros em escala log10.
    """

    vetor_fisico = np.array(
        [
            parametros.R_ext,
            parametros.R_m,
            parametros.C_int,
            parametros.C_m,
            parametros.K_HVAC,
        ],
        dtype=float,
    )

    if not np.all(np.isfinite(vetor_fisico)):
        raise ValueError(
            "Os parâmetros físicos contêm valores "
            "não finitos."
        )

    if np.any(vetor_fisico <= 0):
        raise ValueError(
            "Todos os parâmetros físicos devem ser "
            "estritamente positivos."
        )

    return np.log10(vetor_fisico)


# ============================================================
# Conversão de log10 para parâmetros físicos
# ============================================================


def converter_log10_para_parametros(
    vetor_log10: np.ndarray,
) -> Parametros2R2C:
    """
    Converte um vetor em escala log10 para os parâmetros
    físicos do modelo 2R2C.

    Parameters
    ----------
    vetor_log10 : np.ndarray
        Vetor com dimensão (5,) e ordem:

        [R_ext, R_m, C_int, C_m, K_HVAC]

    Returns
    -------
    Parametros2R2C
        Parâmetros físicos reconstruídos.
    """

    vetor_log10 = validar_vetor_log10(
        vetor_log10
    )

    vetor_fisico = np.power(
        10.0,
        vetor_log10,
    )

    return Parametros2R2C(
        R_ext=float(vetor_fisico[0]),
        R_m=float(vetor_fisico[1]),
        C_int=float(vetor_fisico[2]),
        C_m=float(vetor_fisico[3]),
        K_HVAC=float(vetor_fisico[4]),
    )


# ============================================================
# Validação do vetor logarítmico
# ============================================================


def validar_vetor_log10(
    vetor_log10: np.ndarray,
) -> np.ndarray:
    """
    Valida um vetor de parâmetros em escala log10.

    Parameters
    ----------
    vetor_log10 : np.ndarray
        Vetor dos cinco parâmetros em escala log10.

    Returns
    -------
    np.ndarray
        Vetor convertido para NumPy e tipo float.

    Raises
    ------
    TypeError
        Caso os valores não possam ser convertidos para
        números.

    ValueError
        Caso a dimensão seja incorreta ou existam valores
        não finitos.
    """

    try:
        vetor_log10 = np.asarray(
            vetor_log10,
            dtype=float,
        )
    except (TypeError, ValueError) as erro:
        raise TypeError(
            "O vetor logarítmico deve conter apenas "
            "valores numéricos."
        ) from erro

    if vetor_log10.shape != (5,):
        raise ValueError(
            "O vetor logarítmico deve apresentar "
            "dimensão (5,)."
        )

    if not np.all(np.isfinite(vetor_log10)):
        raise ValueError(
            "O vetor logarítmico contém valores "
            "não finitos."
        )

    return vetor_log10


# ============================================================
# Limites em escala log10
# ============================================================


def obter_limites_log10(
    limites: LimitesIdentificacao2R2C,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Converte os limites físicos da identificação para
    escala log10.

    Parameters
    ----------
    limites : LimitesIdentificacao2R2C
        Limites em unidades físicas.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Limites inferiores e superiores em escala log10.
    """

    validar_limites_identificacao(limites)

    inferiores_log10 = np.log10(
        limites.inferiores
    )

    superiores_log10 = np.log10(
        limites.superiores
    )

    return (
        inferiores_log10,
        superiores_log10,
    )


# ============================================================
# Verificação dos parâmetros face aos limites
# ============================================================


def validar_parametros_nos_limites(
    parametros: Parametros2R2C,
    limites: LimitesIdentificacao2R2C,
) -> None:
    """
    Verifica se os parâmetros físicos estão dentro dos limites
    definidos para a identificação.

    Parameters
    ----------
    parametros : Parametros2R2C
        Parâmetros físicos do modelo.

    limites : LimitesIdentificacao2R2C
        Limites permitidos.
    """

    validar_limites_identificacao(limites)

    vetor_fisico = np.array(
        [
            parametros.R_ext,
            parametros.R_m,
            parametros.C_int,
            parametros.C_m,
            parametros.K_HVAC,
        ],
        dtype=float,
    )

    if np.any(
        vetor_fisico < limites.inferiores
    ):
        raise ValueError(
            "Existem parâmetros inferiores aos limites "
            "admitidos para a identificação."
        )

    if np.any(
        vetor_fisico > limites.superiores
    ):
        raise ValueError(
            "Existem parâmetros superiores aos limites "
            "admitidos para a identificação."
        )


# ============================================================
# Validação da saída experimental
# ============================================================


def validar_saida_medida(
    saida_medida: np.ndarray,
    numero_amostras: int,
) -> np.ndarray:
    """
    Valida a sequência experimental de temperatura interior.

    Parameters
    ----------
    saida_medida : np.ndarray
        Temperatura interior medida.

    numero_amostras : int
        Número esperado de amostras.

    Returns
    -------
    np.ndarray
        Sequência validada com dimensão (N,).
    """

    try:
        saida_medida = np.asarray(
            saida_medida,
            dtype=float,
        )
    except (TypeError, ValueError) as erro:
        raise TypeError(
            "A saída medida deve conter apenas "
            "valores numéricos."
        ) from erro

    if saida_medida.shape != (
        numero_amostras,
    ):
        raise ValueError(
            "A saída medida deve possuir dimensão "
            f"({numero_amostras},)."
        )

    if not np.all(np.isfinite(saida_medida)):
        raise ValueError(
            "A saída medida contém valores não finitos."
        )

    return saida_medida


# ============================================================
# Simulação durante a identificação
# ============================================================


def simular_para_identificacao(
    vetor_log10: np.ndarray,
    entradas: np.ndarray,
    estado_inicial: np.ndarray,
    periodo_amostragem_s: float,
) -> np.ndarray:
    """
    Simula a temperatura interior para um conjunto candidato
    de parâmetros em escala log10.

    O processo executado é:

        vetor log10
            ->
        parâmetros físicos
            ->
        modelo contínuo
            ->
        modelo discreto
            ->
        simulação temporal

    Parameters
    ----------
    vetor_log10 : np.ndarray
        Vetor dos cinco parâmetros em escala log10.

    entradas : np.ndarray
        Matriz de entradas [T_ext, PSum] com dimensão (N, 2).

    estado_inicial : np.ndarray
        Estado inicial [T_int(0), T_m(0)].

    periodo_amostragem_s : float
        Período de amostragem em segundos.

    Returns
    -------
    np.ndarray
        Temperatura interior simulada com dimensão (N,).
    """

    vetor_log10 = validar_vetor_log10(
        vetor_log10
    )

    entradas = validar_entradas_discretas(
        entradas
    )

    estado_inicial = validar_estado_inicial(
        estado_inicial
    )

    periodo_amostragem_s = (
        validar_periodo_amostragem(
            periodo_amostragem_s
        )
    )

    parametros = converter_log10_para_parametros(
        vetor_log10
    )

    matrizes_continuas = (
        construir_matrizes_continuas(
            parametros
        )
    )

    matrizes_discretas = discretizar_modelo(
        matrizes_continuas,
        periodo_amostragem_s,
    )

    resultado = simular_modelo_discreto(
        matrizes=matrizes_discretas,
        entradas=entradas,
        estado_inicial=estado_inicial,
    )

    temperatura_simulada = (
        resultado.saidas[:, 0].copy()
    )

    return temperatura_simulada


# ============================================================
# Cálculo dos resíduos
# ============================================================


def calcular_residuos_identificacao(
    vetor_log10: np.ndarray,
    entradas: np.ndarray,
    saida_medida: np.ndarray,
    estado_inicial: np.ndarray,
    periodo_amostragem_s: float,
) -> np.ndarray:
    """
    Calcula os resíduos entre a temperatura interior medida
    e a temperatura interior simulada pelo modelo.

    A convenção adotada é:

        erro[k] =
            T_int_medida[k] - T_int_simulada[k]

    Parameters
    ----------
    vetor_log10 : np.ndarray
        Vetor dos cinco parâmetros em escala log10.

    entradas : np.ndarray
        Matriz das entradas [T_ext, PSum].

    saida_medida : np.ndarray
        Temperatura interior experimental.

    estado_inicial : np.ndarray
        Estado inicial do modelo.

    periodo_amostragem_s : float
        Período de amostragem em segundos.

    Returns
    -------
    np.ndarray
        Vetor de resíduos com dimensão (N,).
    """

    entradas = validar_entradas_discretas(
        entradas
    )

    numero_amostras = entradas.shape[0]

    saida_medida = validar_saida_medida(
        saida_medida,
        numero_amostras,
    )

    temperatura_simulada = (
        simular_para_identificacao(
            vetor_log10=vetor_log10,
            entradas=entradas,
            estado_inicial=estado_inicial,
            periodo_amostragem_s=periodo_amostragem_s,
        )
    )

    residuos = (
        saida_medida
        - temperatura_simulada
    )

    if residuos.shape != (
        numero_amostras,
    ):
        raise ValueError(
            "O vetor de resíduos apresenta dimensão "
            "incorreta."
        )

    if not np.all(np.isfinite(residuos)):
        raise ValueError(
            "O cálculo dos resíduos produziu valores "
            "não finitos."
        )

    return residuos