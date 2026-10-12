"""
============================================================
Módulo: Identificacao_Coeficientes_2R2C.py

Descrição:
    Funções e estruturas destinadas à identificação direta
    dos coeficientes dinâmicos do modelo térmico 2R2C.

    A parametrização utilizada é:

        a1 = 1 / (R_ext * C_int)

        a2 = 1 / (R_m * C_int)

        a3 = 1 / (R_m * C_m)

        b_HVAC = K_HVAC / C_m

    Esta parametrização permite escrever o modelo contínuo:

        A = [
            [-(a1 + a2),  a2],
            [a3,          -a3]
        ]

        B = [
            [a1,      0],
            [0,   b_HVAC]
        ]

    com:

        x = [T_int, T_m]

        u = [T_ext, P_el]

        y = T_int

    A identificação é realizada diretamente sobre os quatro
    coeficientes dinâmicos, evitando a não unicidade observada
    quando se tenta identificar individualmente:

        R_ext,
        R_m,
        C_int,
        C_m,
        K_HVAC.

    Devido às diferentes ordens de grandeza dos coeficientes,
    o processo de otimização é realizado em escala log10.

Método de otimização:

    scipy.optimize.least_squares

Autor:
    Paulo Ribeiro

Projeto:
    Plataforma de Processamento de Dados Experimentais
    Dissertação de Mestrado
============================================================
"""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares

from Codigos.Modelo_2R2C import (
    validar_entradas_discretas,
    validar_estado_inicial,
    validar_periodo_amostragem,
    discretizar_modelo,
    simular_modelo_discreto,
)

from Codigos.Modelo_2R2C_Coeficientes import (
    Coeficientes2R2C,
    validar_coeficientes,
    construir_matrizes_continuas_coeficientes,
)


# ============================================================
# Estruturas de dados
# ============================================================


@dataclass(frozen=True)
class LimitesCoeficientes2R2C:
    """
    Limites admitidos para os quatro coeficientes dinâmicos.

    A ordem utilizada é:

        [a1, a2, a3, b_HVAC]

    Atributos
    ---------
    inferiores : np.ndarray
        Limites inferiores em unidades físicas.

    superiores : np.ndarray
        Limites superiores em unidades físicas.
    """

    inferiores: np.ndarray
    superiores: np.ndarray


@dataclass(frozen=True)
class MetricasCoeficientes2R2C:
    """
    Métricas utilizadas na avaliação da identificação.

    Atributos
    ---------
    rmse : float
        Raiz do erro quadrático médio [°C].

    mae : float
        Erro absoluto médio [°C].

    erro_medio : float
        Média dos resíduos [°C].

    erro_maximo_absoluto : float
        Maior erro absoluto observado [°C].

    r2 : float
        Coeficiente de determinação.

    custo_quadratico : float
        Metade da soma dos quadrados dos resíduos.
    """

    rmse: float
    mae: float
    erro_medio: float
    erro_maximo_absoluto: float
    r2: float
    custo_quadratico: float


@dataclass(frozen=True)
class ResultadoIdentificacaoCoeficientes2R2C:
    """
    Resultado completo da identificação direta dos
    coeficientes do modelo 2R2C.

    Atributos
    ---------
    vetor_inicial_log10 : np.ndarray
        Ponto inicial da identificação em escala log10.

    vetor_otimizado_log10 : np.ndarray
        Coeficientes identificados em escala log10.

    coeficientes_iniciais : Coeficientes2R2C
        Coeficientes correspondentes ao ponto inicial.

    coeficientes_otimizados : Coeficientes2R2C
        Coeficientes identificados.

    metricas_iniciais : MetricasCoeficientes2R2C
        Métricas antes da identificação.

    metricas_otimizadas : MetricasCoeficientes2R2C
        Métricas após a identificação.

    temperatura_inicial : np.ndarray
        Temperatura simulada com o ponto inicial.

    temperatura_otimizada : np.ndarray
        Temperatura simulada com os coeficientes identificados.

    residuos_iniciais : np.ndarray
        Resíduos antes da identificação.

    residuos_otimizados : np.ndarray
        Resíduos após a identificação.

    sucesso : bool
        Estado da convergência do algoritmo.

    estado : int
        Código de estado devolvido pelo otimizador.

    mensagem : str
        Mensagem devolvida pelo algoritmo.

    numero_avaliacoes : int
        Número de avaliações da função de resíduos.

    melhoria_rmse_percentual : float
        Redução percentual do RMSE.
    """

    vetor_inicial_log10: np.ndarray
    vetor_otimizado_log10: np.ndarray

    coeficientes_iniciais: Coeficientes2R2C
    coeficientes_otimizados: Coeficientes2R2C

    metricas_iniciais: MetricasCoeficientes2R2C
    metricas_otimizadas: MetricasCoeficientes2R2C

    temperatura_inicial: np.ndarray
    temperatura_otimizada: np.ndarray

    residuos_iniciais: np.ndarray
    residuos_otimizados: np.ndarray

    sucesso: bool
    estado: int
    mensagem: str
    numero_avaliacoes: int

    melhoria_rmse_percentual: float


# ============================================================
# Nomes dos coeficientes
# ============================================================


NOMES_COEFICIENTES = (
    "a1",
    "a2",
    "a3",
    "b_HVAC",
)


# ============================================================
# Definição dos limites padrão
# ============================================================


def criar_limites_coeficientes_padrao(
) -> LimitesCoeficientes2R2C:
    """
    Cria limites iniciais para a identificação direta dos
    quatro coeficientes dinâmicos.

    Os limites resultam das gamas físicas anteriormente
    admitidas para os parâmetros R, C e K_HVAC.

    Returns
    -------
    LimitesCoeficientes2R2C
        Limites físicos dos quatro coeficientes.
    """

    inferiores = np.array(
        [
            1.0e-6,   # a1 [1/s]
            1.0e-6,   # a2 [1/s]
            1.0e-8,   # a3 [1/s]
            5.0e-10,  # b_HVAC [K/(W.s)]
        ],
        dtype=float,
    )

    superiores = np.array(
        [
            1.0,      # a1 [1/s]
            1.0,      # a2 [1/s]
            1.0e-2,   # a3 [1/s]
            8.0e-6,   # b_HVAC [K/(W.s)]
        ],
        dtype=float,
    )

    limites = LimitesCoeficientes2R2C(
        inferiores=inferiores,
        superiores=superiores,
    )

    validar_limites_coeficientes(
        limites
    )

    return limites


# ============================================================
# Validação dos limites
# ============================================================


def validar_limites_coeficientes(
    limites: LimitesCoeficientes2R2C,
) -> None:
    """
    Valida os limites utilizados na identificação.

    Parameters
    ----------
    limites : LimitesCoeficientes2R2C
        Limites físicos dos quatro coeficientes.
    """

    if not isinstance(
        limites,
        LimitesCoeficientes2R2C,
    ):
        raise TypeError(
            "Os limites devem ser fornecidos através de uma "
            "instância de LimitesCoeficientes2R2C."
        )

    if limites.inferiores.shape != (4,):
        raise ValueError(
            "Os limites inferiores devem possuir "
            "dimensão (4,)."
        )

    if limites.superiores.shape != (4,):
        raise ValueError(
            "Os limites superiores devem possuir "
            "dimensão (4,)."
        )

    if not np.all(
        np.isfinite(
            limites.inferiores
        )
    ):
        raise ValueError(
            "Os limites inferiores contêm valores "
            "não finitos."
        )

    if not np.all(
        np.isfinite(
            limites.superiores
        )
    ):
        raise ValueError(
            "Os limites superiores contêm valores "
            "não finitos."
        )

    if np.any(
        limites.inferiores <= 0
    ):
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
# Validação do vetor logarítmico
# ============================================================


def validar_vetor_coeficientes_log10(
    vetor_log10: np.ndarray,
) -> np.ndarray:
    """
    Valida um vetor dos quatro coeficientes em escala log10.

    Parameters
    ----------
    vetor_log10 : np.ndarray
        Vetor na ordem:

            [a1, a2, a3, b_HVAC]

    Returns
    -------
    np.ndarray
        Vetor validado.
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

    if vetor_log10.shape != (4,):
        raise ValueError(
            "O vetor logarítmico deve possuir "
            "dimensão (4,)."
        )

    if not np.all(
        np.isfinite(
            vetor_log10
        )
    ):
        raise ValueError(
            "O vetor logarítmico contém valores "
            "não finitos."
        )

    return vetor_log10


# ============================================================
# Conversão coeficientes -> log10
# ============================================================


def converter_coeficientes_para_log10(
    coeficientes: Coeficientes2R2C,
) -> np.ndarray:
    """
    Converte os coeficientes físicos para escala log10.

    Parameters
    ----------
    coeficientes : Coeficientes2R2C
        Coeficientes dinâmicos do modelo.

    Returns
    -------
    np.ndarray
        Vetor:

            [log10(a1),
             log10(a2),
             log10(a3),
             log10(b_HVAC)]
    """

    validar_coeficientes(
        coeficientes
    )

    vetor_fisico = np.array(
        [
            coeficientes.a1,
            coeficientes.a2,
            coeficientes.a3,
            coeficientes.b_HVAC,
        ],
        dtype=float,
    )

    return np.log10(
        vetor_fisico
    )


# ============================================================
# Conversão log10 -> coeficientes
# ============================================================


def converter_log10_para_coeficientes(
    vetor_log10: np.ndarray,
) -> Coeficientes2R2C:
    """
    Converte um vetor em escala log10 para os quatro
    coeficientes dinâmicos físicos.

    Parameters
    ----------
    vetor_log10 : np.ndarray
        Vetor dos quatro coeficientes em escala log10.

    Returns
    -------
    Coeficientes2R2C
        Coeficientes físicos reconstruídos.
    """

    vetor_log10 = (
        validar_vetor_coeficientes_log10(
            vetor_log10
        )
    )

    vetor_fisico = np.power(
        10.0,
        vetor_log10,
    )

    coeficientes = Coeficientes2R2C(
        a1=float(vetor_fisico[0]),
        a2=float(vetor_fisico[1]),
        a3=float(vetor_fisico[2]),
        b_HVAC=float(vetor_fisico[3]),
    )

    validar_coeficientes(
        coeficientes
    )

    return coeficientes


# ============================================================
# Limites em escala log10
# ============================================================


def obter_limites_coeficientes_log10(
    limites: LimitesCoeficientes2R2C,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Converte os limites físicos para escala log10.

    Parameters
    ----------
    limites : LimitesCoeficientes2R2C
        Limites físicos.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Limites inferiores e superiores em log10.
    """

    validar_limites_coeficientes(
        limites
    )

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
# Validação do ponto inicial
# ============================================================


def validar_ponto_inicial_coeficientes(
    vetor_inicial_log10: np.ndarray,
    limites: LimitesCoeficientes2R2C,
) -> np.ndarray:
    """
    Valida um ponto inicial da identificação.
    """

    vetor_inicial_log10 = (
        validar_vetor_coeficientes_log10(
            vetor_inicial_log10
        )
    )

    (
        inferiores_log10,
        superiores_log10,
    ) = obter_limites_coeficientes_log10(
        limites
    )

    if np.any(
        vetor_inicial_log10
        < inferiores_log10
    ):
        raise ValueError(
            "O ponto inicial contém valores inferiores "
            "aos limites definidos."
        )

    if np.any(
        vetor_inicial_log10
        > superiores_log10
    ):
        raise ValueError(
            "O ponto inicial contém valores superiores "
            "aos limites definidos."
        )

    return vetor_inicial_log10


# ============================================================
# Validação da saída medida
# ============================================================


def validar_saida_medida_coeficientes(
    saida_medida: np.ndarray,
    numero_amostras: int,
) -> np.ndarray:
    """
    Valida a sequência experimental de temperatura interior.
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

    if not np.all(
        np.isfinite(
            saida_medida
        )
    ):
        raise ValueError(
            "A saída medida contém valores não finitos."
        )

    return saida_medida


# ============================================================
# Simulação a partir dos coeficientes
# ============================================================


def simular_para_identificacao_coeficientes(
    vetor_log10: np.ndarray,
    entradas: np.ndarray,
    estado_inicial: np.ndarray,
    periodo_amostragem_s: float,
) -> np.ndarray:
    """
    Simula o modelo 2R2C para um vetor candidato dos quatro
    coeficientes em escala log10.

    Parameters
    ----------
    vetor_log10 : np.ndarray
        Coeficientes em escala log10.

    entradas : np.ndarray
        Matriz [T_ext, P_el] com dimensão (N, 2).

    estado_inicial : np.ndarray
        Estado inicial [T_int(0), T_m(0)].

    periodo_amostragem_s : float
        Período de amostragem em segundos.

    Returns
    -------
    np.ndarray
        Temperatura interior simulada com dimensão (N,).
    """

    vetor_log10 = (
        validar_vetor_coeficientes_log10(
            vetor_log10
        )
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

    coeficientes = (
        converter_log10_para_coeficientes(
            vetor_log10
        )
    )

    matrizes_continuas = (
        construir_matrizes_continuas_coeficientes(
            coeficientes
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
# Resíduos
# ============================================================


def calcular_residuos_coeficientes(
    vetor_log10: np.ndarray,
    entradas: np.ndarray,
    saida_medida: np.ndarray,
    estado_inicial: np.ndarray,
    periodo_amostragem_s: float,
) -> np.ndarray:
    """
    Calcula os resíduos da identificação.

    Convenção:

        e[k] =
            T_int_medida[k]
            -
            T_int_simulada[k]
    """

    entradas = validar_entradas_discretas(
        entradas
    )

    numero_amostras = entradas.shape[0]

    saida_medida = (
        validar_saida_medida_coeficientes(
            saida_medida,
            numero_amostras,
        )
    )

    temperatura_simulada = (
        simular_para_identificacao_coeficientes(
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

    if not np.all(
        np.isfinite(
            residuos
        )
    ):
        raise ValueError(
            "Os resíduos contêm valores não finitos."
        )

    return residuos


# ============================================================
# Cálculo das métricas
# ============================================================


def calcular_metricas_coeficientes(
    saida_medida: np.ndarray,
    temperatura_simulada: np.ndarray,
) -> MetricasCoeficientes2R2C:
    """
    Calcula as métricas de ajuste da identificação.
    """

    saida_medida = np.asarray(
        saida_medida,
        dtype=float,
    )

    temperatura_simulada = np.asarray(
        temperatura_simulada,
        dtype=float,
    )

    if saida_medida.shape != temperatura_simulada.shape:
        raise ValueError(
            "A saída medida e a temperatura simulada "
            "devem possuir a mesma dimensão."
        )

    if saida_medida.ndim != 1:
        raise ValueError(
            "As sequências devem ser unidimensionais."
        )

    if saida_medida.size == 0:
        raise ValueError(
            "As sequências não podem estar vazias."
        )

    if not np.all(
        np.isfinite(
            saida_medida
        )
    ):
        raise ValueError(
            "A saída medida contém valores não finitos."
        )

    if not np.all(
        np.isfinite(
            temperatura_simulada
        )
    ):
        raise ValueError(
            "A temperatura simulada contém valores "
            "não finitos."
        )

    residuos = (
        saida_medida
        - temperatura_simulada
    )

    rmse = float(
        np.sqrt(
            np.mean(
                residuos ** 2
            )
        )
    )

    mae = float(
        np.mean(
            np.abs(
                residuos
            )
        )
    )

    erro_medio = float(
        np.mean(
            residuos
        )
    )

    erro_maximo_absoluto = float(
        np.max(
            np.abs(
                residuos
            )
        )
    )

    soma_quadrados_residuos = float(
        np.sum(
            residuos ** 2
        )
    )

    soma_quadrados_total = float(
        np.sum(
            (
                saida_medida
                - np.mean(saida_medida)
            ) ** 2
        )
    )

    if soma_quadrados_total <= 0:
        raise ValueError(
            "Não é possível calcular R² porque a saída "
            "medida não apresenta variabilidade."
        )

    r2 = float(
        1.0
        - soma_quadrados_residuos
        / soma_quadrados_total
    )

    custo_quadratico = float(
        0.5
        * soma_quadrados_residuos
    )

    return MetricasCoeficientes2R2C(
        rmse=rmse,
        mae=mae,
        erro_medio=erro_medio,
        erro_maximo_absoluto=erro_maximo_absoluto,
        r2=r2,
        custo_quadratico=custo_quadratico,
    )


# ============================================================
# Identificação direta dos coeficientes
# ============================================================


def identificar_coeficientes_2r2c(
    vetor_inicial_log10: np.ndarray,
    limites: LimitesCoeficientes2R2C,
    entradas: np.ndarray,
    saida_medida: np.ndarray,
    estado_inicial: np.ndarray,
    periodo_amostragem_s: float,
    maximo_avaliacoes: int = 500,
) -> ResultadoIdentificacaoCoeficientes2R2C:
    """
    Identifica diretamente os quatro coeficientes dinâmicos
    do modelo térmico 2R2C através de mínimos quadrados
    não lineares.

    A otimização é realizada em escala log10.

    Parameters
    ----------
    vetor_inicial_log10 : np.ndarray
        Ponto inicial dos quatro coeficientes.

    limites : LimitesCoeficientes2R2C
        Limites admitidos.

    entradas : np.ndarray
        Entradas [T_ext, P_el].

    saida_medida : np.ndarray
        Temperatura interior experimental.

    estado_inicial : np.ndarray
        Estado inicial do modelo.

    periodo_amostragem_s : float
        Período de amostragem em segundos.

    maximo_avaliacoes : int, default=500
        Máximo de avaliações da função de resíduos.

    Returns
    -------
    ResultadoIdentificacaoCoeficientes2R2C
        Resultado completo da identificação.
    """

    vetor_inicial_log10 = (
        validar_ponto_inicial_coeficientes(
            vetor_inicial_log10,
            limites,
        )
    )

    entradas = validar_entradas_discretas(
        entradas
    )

    numero_amostras = entradas.shape[0]

    saida_medida = (
        validar_saida_medida_coeficientes(
            saida_medida,
            numero_amostras,
        )
    )

    estado_inicial = validar_estado_inicial(
        estado_inicial
    )

    periodo_amostragem_s = (
        validar_periodo_amostragem(
            periodo_amostragem_s
        )
    )

    if not isinstance(
        maximo_avaliacoes,
        (int, np.integer),
    ):
        raise TypeError(
            "O número máximo de avaliações deve ser inteiro."
        )

    maximo_avaliacoes = int(
        maximo_avaliacoes
    )

    if maximo_avaliacoes <= 0:
        raise ValueError(
            "O número máximo de avaliações deve ser "
            "estritamente positivo."
        )

    (
        inferiores_log10,
        superiores_log10,
    ) = obter_limites_coeficientes_log10(
        limites
    )

    # --------------------------------------------------------
    # Simulação inicial
    # --------------------------------------------------------

    coeficientes_iniciais = (
        converter_log10_para_coeficientes(
            vetor_inicial_log10
        )
    )

    temperatura_inicial = (
        simular_para_identificacao_coeficientes(
            vetor_log10=vetor_inicial_log10,
            entradas=entradas,
            estado_inicial=estado_inicial,
            periodo_amostragem_s=periodo_amostragem_s,
        )
    )

    residuos_iniciais = (
        saida_medida
        - temperatura_inicial
    )

    metricas_iniciais = (
        calcular_metricas_coeficientes(
            saida_medida,
            temperatura_inicial,
        )
    )

    # --------------------------------------------------------
    # Otimização
    # --------------------------------------------------------

    resultado_scipy = least_squares(
        fun=calcular_residuos_coeficientes,
        x0=vetor_inicial_log10,
        bounds=(
            inferiores_log10,
            superiores_log10,
        ),
        args=(
            entradas,
            saida_medida,
            estado_inicial,
            periodo_amostragem_s,
        ),
        method="trf",
        ftol=1.0e-10,
        xtol=1.0e-10,
        gtol=1.0e-10,
        max_nfev=maximo_avaliacoes,
    )

    vetor_otimizado_log10 = np.asarray(
        resultado_scipy.x,
        dtype=float,
    )

    vetor_otimizado_log10 = (
        validar_vetor_coeficientes_log10(
            vetor_otimizado_log10
        )
    )

    coeficientes_otimizados = (
        converter_log10_para_coeficientes(
            vetor_otimizado_log10
        )
    )

    # --------------------------------------------------------
    # Simulação final
    # --------------------------------------------------------

    temperatura_otimizada = (
        simular_para_identificacao_coeficientes(
            vetor_log10=vetor_otimizado_log10,
            entradas=entradas,
            estado_inicial=estado_inicial,
            periodo_amostragem_s=periodo_amostragem_s,
        )
    )

    residuos_otimizados = (
        saida_medida
        - temperatura_otimizada
    )

    metricas_otimizadas = (
        calcular_metricas_coeficientes(
            saida_medida,
            temperatura_otimizada,
        )
    )

    # --------------------------------------------------------
    # Melhoria do RMSE
    # --------------------------------------------------------

    if metricas_iniciais.rmse <= 0:

        melhoria_rmse_percentual = 0.0

    else:

        melhoria_rmse_percentual = float(
            (
                metricas_iniciais.rmse
                - metricas_otimizadas.rmse
            )
            / metricas_iniciais.rmse
            * 100.0
        )

    return ResultadoIdentificacaoCoeficientes2R2C(
        vetor_inicial_log10=(
            vetor_inicial_log10.copy()
        ),
        vetor_otimizado_log10=(
            vetor_otimizado_log10.copy()
        ),
        coeficientes_iniciais=(
            coeficientes_iniciais
        ),
        coeficientes_otimizados=(
            coeficientes_otimizados
        ),
        metricas_iniciais=(
            metricas_iniciais
        ),
        metricas_otimizadas=(
            metricas_otimizadas
        ),
        temperatura_inicial=(
            temperatura_inicial.copy()
        ),
        temperatura_otimizada=(
            temperatura_otimizada.copy()
        ),
        residuos_iniciais=(
            residuos_iniciais.copy()
        ),
        residuos_otimizados=(
            residuos_otimizados.copy()
        ),
        sucesso=bool(
            resultado_scipy.success
        ),
        estado=int(
            resultado_scipy.status
        ),
        mensagem=str(
            resultado_scipy.message
        ),
        numero_avaliacoes=int(
            resultado_scipy.nfev
        ),
        melhoria_rmse_percentual=(
            melhoria_rmse_percentual
        ),
    )


# ============================================================
# Validação do resultado
# ============================================================


def validar_resultado_identificacao_coeficientes(
    resultado: ResultadoIdentificacaoCoeficientes2R2C,
) -> None:
    """
    Valida estruturalmente o resultado da identificação.
    """

    if not isinstance(
        resultado,
        ResultadoIdentificacaoCoeficientes2R2C,
    ):
        raise TypeError(
            "O resultado deve ser uma instância de "
            "ResultadoIdentificacaoCoeficientes2R2C."
        )

    validar_coeficientes(
        resultado.coeficientes_iniciais
    )

    validar_coeficientes(
        resultado.coeficientes_otimizados
    )

    if (
        resultado.temperatura_inicial.shape
        != resultado.temperatura_otimizada.shape
    ):
        raise ValueError(
            "As temperaturas inicial e otimizada devem "
            "possuir a mesma dimensão."
        )

    if (
        resultado.residuos_iniciais.shape
        != resultado.residuos_otimizados.shape
    ):
        raise ValueError(
            "Os resíduos iniciais e otimizados devem "
            "possuir a mesma dimensão."
        )

    if not np.all(
        np.isfinite(
            resultado.temperatura_otimizada
        )
    ):
        raise ValueError(
            "A temperatura otimizada contém valores "
            "não finitos."
        )

    if not np.all(
        np.isfinite(
            resultado.residuos_otimizados
        )
    ):
        raise ValueError(
            "Os resíduos otimizados contêm valores "
            "não finitos."
        )


# ============================================================
# Apresentação dos resultados
# ============================================================


def mostrar_resultado_identificacao_coeficientes(
    resultado: ResultadoIdentificacaoCoeficientes2R2C,
) -> None:
    """
    Apresenta um resumo da identificação dos coeficientes.
    """

    validar_resultado_identificacao_coeficientes(
        resultado
    )

    inicial = (
        resultado.coeficientes_iniciais
    )

    final = (
        resultado.coeficientes_otimizados
    )

    metricas_i = (
        resultado.metricas_iniciais
    )

    metricas_f = (
        resultado.metricas_otimizadas
    )

    print(
        "========== IDENTIFICAÇÃO DIRETA DOS "
        "COEFICIENTES 2R2C ==========\n"
    )

    print(
        f"Convergência: "
        f"{resultado.sucesso}"
    )

    print(
        f"Estado do otimizador: "
        f"{resultado.estado}"
    )

    print(
        f"Número de avaliações: "
        f"{resultado.numero_avaliacoes}"
    )

    print(
        f"Mensagem: "
        f"{resultado.mensagem}"
    )

    print(
        "\nCoeficientes iniciais:"
    )

    print(
        f"a1     = "
        f"{inicial.a1:.12e} 1/s"
    )

    print(
        f"a2     = "
        f"{inicial.a2:.12e} 1/s"
    )

    print(
        f"a3     = "
        f"{inicial.a3:.12e} 1/s"
    )

    print(
        f"b_HVAC = "
        f"{inicial.b_HVAC:.12e} K/(W.s)"
    )

    print(
        "\nCoeficientes identificados:"
    )

    print(
        f"a1     = "
        f"{final.a1:.12e} 1/s"
    )

    print(
        f"a2     = "
        f"{final.a2:.12e} 1/s"
    )

    print(
        f"a3     = "
        f"{final.a3:.12e} 1/s"
    )

    print(
        f"b_HVAC = "
        f"{final.b_HVAC:.12e} K/(W.s)"
    )

    print(
        "\nMétricas iniciais:"
    )

    print(
        f"RMSE = "
        f"{metricas_i.rmse:.6f} °C"
    )

    print(
        f"MAE = "
        f"{metricas_i.mae:.6f} °C"
    )

    print(
        f"R² = "
        f"{metricas_i.r2:.6f}"
    )

    print(
        "\nMétricas após identificação:"
    )

    print(
        f"RMSE = "
        f"{metricas_f.rmse:.6f} °C"
    )

    print(
        f"MAE = "
        f"{metricas_f.mae:.6f} °C"
    )

    print(
        f"Erro médio = "
        f"{metricas_f.erro_medio:.6f} °C"
    )

    print(
        f"Erro máximo absoluto = "
        f"{metricas_f.erro_maximo_absoluto:.6f} °C"
    )

    print(
        f"R² = "
        f"{metricas_f.r2:.6f}"
    )

    print(
        "\nMelhoria do RMSE:"
    )

    print(
        f"{resultado.melhoria_rmse_percentual:.2f} %"
    )

    print(
        "\n====================================================="
    )