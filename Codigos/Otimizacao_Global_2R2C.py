"""
============================================================
Módulo: Otimizacao_Global_2R2C.py

Descrição:
    Funções e estruturas destinadas à otimização global dos
    parâmetros do modelo térmico 2R2C.

    O módulo utiliza o problema de identificação definido em
    Identificacao_2R2C.py e aplica o algoritmo de evolução
    diferencial disponibilizado por:

        scipy.optimize.differential_evolution

    Esta abordagem constitui uma metodologia independente da
    otimização local baseada em least_squares, permitindo
    posteriormente comparar:

        - qualidade do ajuste;
        - capacidade de generalização;
        - parâmetros identificados;
        - proximidade aos limites;
        - robustez da solução;
        - esforço computacional.

Parâmetros identificados:

    theta = [
        R_ext,
        R_m,
        C_int,
        C_m,
        K_HVAC
    ]

Representação utilizada pelo otimizador:

    p = log10(theta)

Função objetivo:

    J(p) = sum(e[k]^2)

onde:

    e[k] =
        T_int_medida[k]
        -
        T_int_simulada[k]

Autor:
    Paulo Ribeiro

Projeto:
    Plataforma de Processamento de Dados Experimentais
    Dissertação de Mestrado
============================================================
"""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import differential_evolution

from Codigos.Modelo_2R2C import (
    Parametros2R2C,
    validar_estado_inicial,
    validar_entradas_discretas,
    validar_periodo_amostragem,
)

from Codigos.Identificacao_2R2C import (
    LimitesIdentificacao2R2C,
    validar_limites_identificacao,
    validar_saida_medida,
    validar_vetor_log10,
    obter_limites_log10,
    converter_log10_para_parametros,
    validar_parametros_nos_limites,
    simular_para_identificacao,
    calcular_residuos_identificacao,
)

from Codigos.Otimizacao_2R2C import (
    MetricasAjuste2R2C,
    calcular_metricas_ajuste,
    identificar_parametros_proximos_limites,
)


# ============================================================
# Estruturas de dados
# ============================================================


@dataclass(frozen=True)
class ResultadoOtimizacaoGlobal2R2C:
    """
    Resultado completo da otimização global do modelo 2R2C.

    Atributos
    ---------
    vetor_otimizado_log10 : np.ndarray
        Melhor vetor encontrado em escala log10.

    parametros_otimizados : Parametros2R2C
        Parâmetros físicos correspondentes à melhor solução.

    metricas_otimizadas : MetricasAjuste2R2C
        Métricas calculadas com a solução otimizada.

    temperatura_otimizada : np.ndarray
        Temperatura interior simulada com os parâmetros
        otimizados.

    residuos_otimizados : np.ndarray
        Resíduos correspondentes à solução otimizada.

    sucesso : bool
        Indica se o algoritmo terminou com sucesso.

    mensagem : str
        Mensagem devolvida pelo otimizador.

    numero_iteracoes : int
        Número de gerações executadas.

    numero_avaliacoes : int
        Número de avaliações da função objetivo.

    valor_funcao_objetivo : float
        Valor final da soma dos quadrados dos resíduos.

    parametros_proximos_limites : tuple[str, ...]
        Parâmetros identificados próximos dos limites.

    semente_aleatoria : int
        Semente utilizada pelo algoritmo.

    populacao_relativa : int
        Fator de dimensão da população do algoritmo.
    """

    vetor_otimizado_log10: np.ndarray

    parametros_otimizados: Parametros2R2C

    metricas_otimizadas: MetricasAjuste2R2C

    temperatura_otimizada: np.ndarray

    residuos_otimizados: np.ndarray

    sucesso: bool

    mensagem: str

    numero_iteracoes: int

    numero_avaliacoes: int

    valor_funcao_objetivo: float

    parametros_proximos_limites: tuple[str, ...]

    semente_aleatoria: int

    populacao_relativa: int


# ============================================================
# Função objetivo
# ============================================================


def calcular_funcao_objetivo_global(
    vetor_log10: np.ndarray,
    entradas: np.ndarray,
    saida_medida: np.ndarray,
    estado_inicial: np.ndarray,
    periodo_amostragem_s: float,
) -> float:
    """
    Calcula a função objetivo da otimização global.

    A função objetivo corresponde à soma dos quadrados dos
    resíduos:

        J =
            sum(
                (
                    T_int_medida
                    -
                    T_int_simulada
                ) ** 2
            )

    Parameters
    ----------
    vetor_log10 : np.ndarray
        Vetor dos cinco parâmetros em escala log10.

    entradas : np.ndarray
        Entradas experimentais [T_ext, PSum].

    saida_medida : np.ndarray
        Temperatura interior experimental.

    estado_inicial : np.ndarray
        Estado inicial do modelo.

    periodo_amostragem_s : float
        Período de amostragem em segundos.

    Returns
    -------
    float
        Valor escalar da função objetivo.
    """

    residuos = calcular_residuos_identificacao(
        vetor_log10=vetor_log10,
        entradas=entradas,
        saida_medida=saida_medida,
        estado_inicial=estado_inicial,
        periodo_amostragem_s=periodo_amostragem_s,
    )

    valor_objetivo = float(
        np.sum(
            residuos ** 2
        )
    )

    if not np.isfinite(
        valor_objetivo
    ):
        raise ValueError(
            "A função objetivo produziu um valor não finito."
        )

    return valor_objetivo


# ============================================================
# Construção dos limites para differential_evolution
# ============================================================


def construir_limites_otimizacao_global(
    limites: LimitesIdentificacao2R2C,
) -> list[tuple[float, float]]:
    """
    Converte os limites físicos para o formato utilizado por
    scipy.optimize.differential_evolution.

    Parameters
    ----------
    limites : LimitesIdentificacao2R2C
        Limites físicos da identificação.

    Returns
    -------
    list[tuple[float, float]]
        Lista de cinco pares:

            [
                (lim_inf_1, lim_sup_1),
                ...
            ]

        expressos em escala log10.
    """

    validar_limites_identificacao(
        limites
    )

    (
        inferiores_log10,
        superiores_log10,
    ) = obter_limites_log10(
        limites
    )

    limites_global = [
        (
            float(inferiores_log10[indice]),
            float(superiores_log10[indice]),
        )
        for indice in range(5)
    ]

    return limites_global


# ============================================================
# Otimização global
# ============================================================


def otimizar_parametros_global_2r2c(
    limites: LimitesIdentificacao2R2C,
    entradas: np.ndarray,
    saida_medida: np.ndarray,
    estado_inicial: np.ndarray,
    periodo_amostragem_s: float,
    maximo_iteracoes: int = 100,
    populacao_relativa: int = 10,
    tolerancia: float = 1.0e-6,
    semente_aleatoria: int = 42,
    polimento_final: bool = True,
) -> ResultadoOtimizacaoGlobal2R2C:
    """
    Executa a identificação global dos parâmetros do modelo
    térmico 2R2C através de evolução diferencial.

    Parameters
    ----------
    limites : LimitesIdentificacao2R2C
        Limites físicos dos cinco parâmetros.

    entradas : np.ndarray
        Matriz de entradas [T_ext, PSum].

    saida_medida : np.ndarray
        Temperatura interior experimental.

    estado_inicial : np.ndarray
        Estado inicial [T_int(0), T_m(0)].

    periodo_amostragem_s : float
        Período de amostragem em segundos.

    maximo_iteracoes : int, default=100
        Número máximo de gerações do algoritmo.

    populacao_relativa : int, default=10
        Multiplicador utilizado na dimensão da população.

    tolerancia : float, default=1e-6
        Tolerância de convergência.

    semente_aleatoria : int, default=42
        Semente utilizada para garantir reprodutibilidade.

    polimento_final : bool, default=True
        Quando True, scipy aplica um refinamento local à
        melhor solução global encontrada.

    Returns
    -------
    ResultadoOtimizacaoGlobal2R2C
        Resultado completo da otimização global.
    """

    validar_limites_identificacao(
        limites
    )

    entradas = validar_entradas_discretas(
        entradas
    )

    numero_amostras = entradas.shape[0]

    saida_medida = validar_saida_medida(
        saida_medida,
        numero_amostras,
    )

    estado_inicial = validar_estado_inicial(
        estado_inicial
    )

    periodo_amostragem_s = validar_periodo_amostragem(
        periodo_amostragem_s
    )

    if not isinstance(
        maximo_iteracoes,
        (int, np.integer),
    ):
        raise TypeError(
            "O número máximo de iterações deve ser inteiro."
        )

    maximo_iteracoes = int(
        maximo_iteracoes
    )

    if maximo_iteracoes <= 0:
        raise ValueError(
            "O número máximo de iterações deve ser "
            "estritamente positivo."
        )

    if not isinstance(
        populacao_relativa,
        (int, np.integer),
    ):
        raise TypeError(
            "A população relativa deve ser inteira."
        )

    populacao_relativa = int(
        populacao_relativa
    )

    if populacao_relativa <= 0:
        raise ValueError(
            "A população relativa deve ser "
            "estritamente positiva."
        )

    tolerancia = float(
        tolerancia
    )

    if (
        not np.isfinite(tolerancia)
        or tolerancia <= 0
    ):
        raise ValueError(
            "A tolerância deve ser finita e "
            "estritamente positiva."
        )

    if not isinstance(
        semente_aleatoria,
        (int, np.integer),
    ):
        raise TypeError(
            "A semente aleatória deve ser inteira."
        )

    semente_aleatoria = int(
        semente_aleatoria
    )

    limites_global = (
        construir_limites_otimizacao_global(
            limites
        )
    )

    resultado_scipy = differential_evolution(
        func=calcular_funcao_objetivo_global,
        bounds=limites_global,
        args=(
            entradas,
            saida_medida,
            estado_inicial,
            periodo_amostragem_s,
        ),
        strategy="best1bin",
        maxiter=maximo_iteracoes,
        popsize=populacao_relativa,
        tol=tolerancia,
        mutation=(
            0.5,
            1.0,
        ),
        recombination=0.7,
        seed=semente_aleatoria,
        polish=bool(
            polimento_final
        ),
        updating="immediate",
        workers=1,
        disp=False,
    )

    vetor_otimizado_log10 = np.asarray(
        resultado_scipy.x,
        dtype=float,
    )

    validar_vetor_log10(
        vetor_otimizado_log10
    )

    parametros_otimizados = (
        converter_log10_para_parametros(
            vetor_otimizado_log10
        )
    )

    validar_parametros_nos_limites(
        parametros_otimizados,
        limites,
    )

    temperatura_otimizada = (
        simular_para_identificacao(
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
        calcular_metricas_ajuste(
            saida_medida=saida_medida,
            temperatura_simulada=temperatura_otimizada,
        )
    )

    parametros_proximos_limites = (
        identificar_parametros_proximos_limites(
            vetor_otimizado_log10,
            limites,
        )
    )

    return ResultadoOtimizacaoGlobal2R2C(
        vetor_otimizado_log10=(
            vetor_otimizado_log10.copy()
        ),
        parametros_otimizados=(
            parametros_otimizados
        ),
        metricas_otimizadas=(
            metricas_otimizadas
        ),
        temperatura_otimizada=(
            temperatura_otimizada.copy()
        ),
        residuos_otimizados=(
            residuos_otimizados.copy()
        ),
        sucesso=bool(
            resultado_scipy.success
        ),
        mensagem=str(
            resultado_scipy.message
        ),
        numero_iteracoes=int(
            resultado_scipy.nit
        ),
        numero_avaliacoes=int(
            resultado_scipy.nfev
        ),
        valor_funcao_objetivo=float(
            resultado_scipy.fun
        ),
        parametros_proximos_limites=(
            parametros_proximos_limites
        ),
        semente_aleatoria=(
            semente_aleatoria
        ),
        populacao_relativa=(
            populacao_relativa
        ),
    )


# ============================================================
# Validação do resultado
# ============================================================


def validar_resultado_otimizacao_global(
    resultado: ResultadoOtimizacaoGlobal2R2C,
) -> None:
    """
    Valida estruturalmente o resultado da otimização global.

    Parameters
    ----------
    resultado : ResultadoOtimizacaoGlobal2R2C
        Resultado a validar.
    """

    if not isinstance(
        resultado,
        ResultadoOtimizacaoGlobal2R2C,
    ):
        raise TypeError(
            "O resultado deve ser uma instância de "
            "ResultadoOtimizacaoGlobal2R2C."
        )

    validar_vetor_log10(
        resultado.vetor_otimizado_log10
    )

    if resultado.temperatura_otimizada.ndim != 1:
        raise ValueError(
            "A temperatura otimizada deve ser "
            "unidimensional."
        )

    if resultado.residuos_otimizados.ndim != 1:
        raise ValueError(
            "Os resíduos otimizados devem ser "
            "unidimensionais."
        )

    if (
        resultado.temperatura_otimizada.shape
        != resultado.residuos_otimizados.shape
    ):
        raise ValueError(
            "A temperatura otimizada e os resíduos "
            "devem possuir a mesma dimensão."
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

    if not np.isfinite(
        resultado.valor_funcao_objetivo
    ):
        raise ValueError(
            "O valor da função objetivo não é finito."
        )


# ============================================================
# Apresentação dos resultados
# ============================================================


def mostrar_resultado_otimizacao_global(
    resultado: ResultadoOtimizacaoGlobal2R2C,
) -> None:
    """
    Apresenta o resumo da otimização global.

    Parameters
    ----------
    resultado : ResultadoOtimizacaoGlobal2R2C
        Resultado da otimização global.
    """

    validar_resultado_otimizacao_global(
        resultado
    )

    parametros = (
        resultado.parametros_otimizados
    )

    metricas = (
        resultado.metricas_otimizadas
    )

    print(
        "========== OTIMIZAÇÃO GLOBAL DO MODELO 2R2C ==========\n"
    )

    print(
        f"Convergência: "
        f"{resultado.sucesso}"
    )

    print(
        f"Mensagem: "
        f"{resultado.mensagem}"
    )

    print(
        f"Número de iterações: "
        f"{resultado.numero_iteracoes}"
    )

    print(
        f"Número de avaliações: "
        f"{resultado.numero_avaliacoes}"
    )

    print(
        f"Semente aleatória: "
        f"{resultado.semente_aleatoria}"
    )

    print(
        f"População relativa: "
        f"{resultado.populacao_relativa}"
    )

    print(
        "\nParâmetros otimizados:"
    )

    print(
        f"R_ext  = "
        f"{parametros.R_ext:.6e} K/W"
    )

    print(
        f"R_m    = "
        f"{parametros.R_m:.6e} K/W"
    )

    print(
        f"C_int  = "
        f"{parametros.C_int:.6e} J/K"
    )

    print(
        f"C_m    = "
        f"{parametros.C_m:.6e} J/K"
    )

    print(
        f"K_HVAC = "
        f"{parametros.K_HVAC:.6f}"
    )

    print(
        "\nMétricas:"
    )

    print(
        f"RMSE = "
        f"{metricas.rmse:.6f} °C"
    )

    print(
        f"MAE = "
        f"{metricas.mae:.6f} °C"
    )

    print(
        f"Erro médio = "
        f"{metricas.erro_medio:.6f} °C"
    )

    print(
        f"Erro máximo absoluto = "
        f"{metricas.erro_maximo_absoluto:.6f} °C"
    )

    print(
        f"R² = "
        f"{metricas.r2:.6f}"
    )

    print(
        "\nFunção objetivo:"
    )

    print(
        f"{resultado.valor_funcao_objetivo:.12e}"
    )

    print(
        "\nParâmetros próximos dos limites:"
    )

    if resultado.parametros_proximos_limites:

        for nome in (
            resultado.parametros_proximos_limites
        ):
            print(
                f"  - {nome}"
            )

    else:

        print(
            "Nenhum."
        )

    print(
        "\n======================================================="
    )