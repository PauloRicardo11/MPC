"""
============================================================
Módulo: Otimizacao_2R2C.py

Descrição:
    Funções e estruturas destinadas à otimização dos
    parâmetros do modelo térmico 2R2C.

    O módulo utiliza o problema de identificação definido
    em Identificacao_2R2C.py e aplica o algoritmo de mínimos
    quadrados não lineares disponível em scipy.optimize.

    Nesta primeira versão, o módulo permite:

    - validar um ponto inicial de otimização;
    - executar uma identificação com uma única inicialização;
    - respeitar os limites físicos definidos para os
      parâmetros;
    - trabalhar em escala logarítmica;
    - calcular métricas antes e depois da otimização;
    - avaliar a melhoria obtida;
    - verificar a proximidade dos parâmetros aos limites;
    - apresentar um resumo completo da otimização.

    A estratégia multistart será acrescentada apenas depois
    de esta primeira versão estar validada.

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

Método:

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
    Parametros2R2C,
    validar_estado_inicial,
    validar_entradas_discretas,
    validar_periodo_amostragem,
)

from Codigos.Identificacao_2R2C import (
    LimitesIdentificacao2R2C,
    validar_limites_identificacao,
    validar_vetor_log10,
    validar_saida_medida,
    obter_limites_log10,
    converter_log10_para_parametros,
    validar_parametros_nos_limites,
    simular_para_identificacao,
    calcular_residuos_identificacao,
)


# ============================================================
# Estruturas de dados
# ============================================================


@dataclass(frozen=True)
class MetricasAjuste2R2C:
    """
    Métricas utilizadas na avaliação do ajuste do modelo.

    Atributos
    ---------
    rmse : float
        Raiz do erro quadrático médio.

    mae : float
        Erro absoluto médio.

    erro_medio : float
        Média dos resíduos.

    erro_maximo_absoluto : float
        Maior erro absoluto observado.

    r2 : float
        Coeficiente de determinação.

    custo_quadratico : float
        Metade da soma dos quadrados dos resíduos, de acordo
        com a convenção utilizada por scipy.optimize.
    """

    rmse: float
    mae: float
    erro_medio: float
    erro_maximo_absoluto: float
    r2: float
    custo_quadratico: float


@dataclass(frozen=True)
class ResultadoOtimizacao2R2C:
    """
    Resultado completo de uma otimização dos parâmetros 2R2C.

    Atributos
    ---------
    vetor_inicial_log10 : np.ndarray
        Ponto inicial utilizado pelo otimizador.

    vetor_otimizado_log10 : np.ndarray
        Melhor vetor encontrado em escala log10.

    parametros_iniciais : Parametros2R2C
        Parâmetros físicos correspondentes ao ponto inicial.

    parametros_otimizados : Parametros2R2C
        Parâmetros físicos obtidos após otimização.

    metricas_iniciais : MetricasAjuste2R2C
        Métricas antes da otimização.

    metricas_otimizadas : MetricasAjuste2R2C
        Métricas após a otimização.

    temperatura_inicial : np.ndarray
        Temperatura simulada antes da otimização.

    temperatura_otimizada : np.ndarray
        Temperatura simulada após a otimização.

    residuos_iniciais : np.ndarray
        Resíduos antes da otimização.

    residuos_otimizados : np.ndarray
        Resíduos após a otimização.

    sucesso : bool
        Indica se o algoritmo terminou com sucesso.

    estado : int
        Código de estado devolvido pelo otimizador.

    mensagem : str
        Mensagem devolvida pelo algoritmo.

    numero_avaliacoes : int
        Número de avaliações da função de resíduos.

    melhoria_rmse_percentual : float
        Melhoria percentual do RMSE.

    parametros_proximos_limites : tuple[str, ...]
        Nomes dos parâmetros considerados demasiado próximos
        de um dos limites da região de procura.
    """

    vetor_inicial_log10: np.ndarray
    vetor_otimizado_log10: np.ndarray

    parametros_iniciais: Parametros2R2C
    parametros_otimizados: Parametros2R2C

    metricas_iniciais: MetricasAjuste2R2C
    metricas_otimizadas: MetricasAjuste2R2C

    temperatura_inicial: np.ndarray
    temperatura_otimizada: np.ndarray

    residuos_iniciais: np.ndarray
    residuos_otimizados: np.ndarray

    sucesso: bool
    estado: int
    mensagem: str
    numero_avaliacoes: int

    melhoria_rmse_percentual: float

    parametros_proximos_limites: tuple[str, ...]


# ============================================================
# Nomes dos parâmetros
# ============================================================


NOMES_PARAMETROS = (
    "R_ext",
    "R_m",
    "C_int",
    "C_m",
    "K_HVAC",
)


# ============================================================
# Validação do ponto inicial
# ============================================================


def validar_ponto_inicial(
    vetor_inicial_log10: np.ndarray,
    limites: LimitesIdentificacao2R2C,
) -> np.ndarray:
    """
    Valida o ponto inicial utilizado na otimização.

    O vetor deve:

    - possuir cinco elementos;
    - conter apenas valores finitos;
    - estar dentro dos limites logarítmicos definidos.

    Parameters
    ----------
    vetor_inicial_log10 : np.ndarray
        Ponto inicial em escala log10.

    limites : LimitesIdentificacao2R2C
        Limites físicos admissíveis.

    Returns
    -------
    np.ndarray
        Ponto inicial validado.
    """

    vetor_inicial_log10 = validar_vetor_log10(
        vetor_inicial_log10
    )

    validar_limites_identificacao(
        limites
    )

    (
        limites_inferiores_log10,
        limites_superiores_log10,
    ) = obter_limites_log10(
        limites
    )

    if np.any(
        vetor_inicial_log10
        < limites_inferiores_log10
    ):
        raise ValueError(
            "O ponto inicial contém valores inferiores "
            "aos limites definidos."
        )

    if np.any(
        vetor_inicial_log10
        > limites_superiores_log10
    ):
        raise ValueError(
            "O ponto inicial contém valores superiores "
            "aos limites definidos."
        )

    parametros_iniciais = (
        converter_log10_para_parametros(
            vetor_inicial_log10
        )
    )

    validar_parametros_nos_limites(
        parametros_iniciais,
        limites,
    )

    return vetor_inicial_log10


# ============================================================
# Cálculo das métricas
# ============================================================


def calcular_metricas_ajuste(
    saida_medida: np.ndarray,
    temperatura_simulada: np.ndarray,
) -> MetricasAjuste2R2C:
    """
    Calcula as principais métricas de qualidade do ajuste.

    A convenção dos resíduos é:

        e[k] =
            T_int_medida[k]
            - T_int_simulada[k]

    Parameters
    ----------
    saida_medida : np.ndarray
        Temperatura interior experimental.

    temperatura_simulada : np.ndarray
        Temperatura interior prevista pelo modelo.

    Returns
    -------
    MetricasAjuste2R2C
        Métricas calculadas.
    """

    try:
        saida_medida = np.asarray(
            saida_medida,
            dtype=float,
        )

        temperatura_simulada = np.asarray(
            temperatura_simulada,
            dtype=float,
        )

    except (TypeError, ValueError) as erro:
        raise TypeError(
            "As temperaturas devem conter apenas "
            "valores numéricos."
        ) from erro

    if saida_medida.ndim != 1:
        raise ValueError(
            "A saída medida deve ser unidimensional."
        )

    if temperatura_simulada.ndim != 1:
        raise ValueError(
            "A temperatura simulada deve ser "
            "unidimensional."
        )

    if saida_medida.shape != temperatura_simulada.shape:
        raise ValueError(
            "A saída medida e a temperatura simulada "
            "devem possuir a mesma dimensão."
        )

    if saida_medida.size == 0:
        raise ValueError(
            "As sequências de temperatura não podem "
            "estar vazias."
        )

    if not np.all(
        np.isfinite(saida_medida)
    ):
        raise ValueError(
            "A saída medida contém valores não finitos."
        )

    if not np.all(
        np.isfinite(temperatura_simulada)
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

    return MetricasAjuste2R2C(
        rmse=rmse,
        mae=mae,
        erro_medio=erro_medio,
        erro_maximo_absoluto=erro_maximo_absoluto,
        r2=r2,
        custo_quadratico=custo_quadratico,
    )


# ============================================================
# Verificação da proximidade aos limites
# ============================================================


def identificar_parametros_proximos_limites(
    vetor_log10: np.ndarray,
    limites: LimitesIdentificacao2R2C,
    tolerancia_relativa: float = 0.02,
) -> tuple[str, ...]:
    """
    Identifica parâmetros demasiado próximos dos limites.

    A avaliação é realizada no espaço logarítmico.

    Um parâmetro é considerado próximo de um limite quando
    está dentro da fração definida por tolerancia_relativa
    relativamente à amplitude total do intervalo logarítmico.

    Parameters
    ----------
    vetor_log10 : np.ndarray
        Vetor de parâmetros em escala log10.

    limites : LimitesIdentificacao2R2C
        Limites da identificação.

    tolerancia_relativa : float, default=0.02
        Fração da amplitude considerada próxima de um limite.

    Returns
    -------
    tuple[str, ...]
        Nomes dos parâmetros próximos dos limites.
    """

    vetor_log10 = validar_vetor_log10(
        vetor_log10
    )

    validar_limites_identificacao(
        limites
    )

    if not isinstance(
        tolerancia_relativa,
        (int, float, np.integer, np.floating),
    ):
        raise TypeError(
            "A tolerância relativa deve ser numérica."
        )

    tolerancia_relativa = float(
        tolerancia_relativa
    )

    if (
        tolerancia_relativa <= 0
        or tolerancia_relativa >= 0.5
    ):
        raise ValueError(
            "A tolerância relativa deve estar "
            "entre 0 e 0.5."
        )

    (
        inferiores_log10,
        superiores_log10,
    ) = obter_limites_log10(
        limites
    )

    amplitudes = (
        superiores_log10
        - inferiores_log10
    )

    distancias_inferiores = (
        vetor_log10
        - inferiores_log10
    )

    distancias_superiores = (
        superiores_log10
        - vetor_log10
    )

    limite_proximidade = (
        tolerancia_relativa
        * amplitudes
    )

    parametros_proximos = []

    for indice, nome in enumerate(
        NOMES_PARAMETROS
    ):

        if (
            distancias_inferiores[indice]
            <= limite_proximidade[indice]
            or
            distancias_superiores[indice]
            <= limite_proximidade[indice]
        ):
            parametros_proximos.append(
                nome
            )

    return tuple(
        parametros_proximos
    )


# ============================================================
# Cálculo da melhoria percentual
# ============================================================


def calcular_melhoria_rmse_percentual(
    rmse_inicial: float,
    rmse_final: float,
) -> float:
    """
    Calcula a melhoria percentual do RMSE.

    Parameters
    ----------
    rmse_inicial : float
        RMSE antes da otimização.

    rmse_final : float
        RMSE depois da otimização.

    Returns
    -------
    float
        Melhoria percentual.
    """

    rmse_inicial = float(
        rmse_inicial
    )

    rmse_final = float(
        rmse_final
    )

    if not np.isfinite(
        rmse_inicial
    ):
        raise ValueError(
            "O RMSE inicial deve ser finito."
        )

    if not np.isfinite(
        rmse_final
    ):
        raise ValueError(
            "O RMSE final deve ser finito."
        )

    if rmse_inicial <= 0:
        raise ValueError(
            "O RMSE inicial deve ser superior a zero."
        )

    melhoria = (
        (
            rmse_inicial
            - rmse_final
        )
        / rmse_inicial
        * 100.0
    )

    return float(
        melhoria
    )


# ============================================================
# Otimização dos parâmetros
# ============================================================


def otimizar_parametros_2r2c(
    vetor_inicial_log10: np.ndarray,
    limites: LimitesIdentificacao2R2C,
    entradas: np.ndarray,
    saida_medida: np.ndarray,
    estado_inicial: np.ndarray,
    periodo_amostragem_s: float,
    maximo_avaliacoes: int = 500,
) -> ResultadoOtimizacao2R2C:
    """
    Executa uma otimização dos cinco parâmetros do modelo 2R2C.

    A identificação é realizada através de mínimos quadrados
    não lineares no espaço logarítmico:

        p = log10(theta)

    utilizando scipy.optimize.least_squares.

    Nesta primeira versão é utilizada uma única inicialização.

    Parameters
    ----------
    vetor_inicial_log10 : np.ndarray
        Ponto inicial da otimização.

    limites : LimitesIdentificacao2R2C
        Limites físicos dos parâmetros.

    entradas : np.ndarray
        Entradas experimentais [T_ext, PSum].

    saida_medida : np.ndarray
        Temperatura interior experimental.

    estado_inicial : np.ndarray
        Estado inicial [T_int(0), T_m(0)].

    periodo_amostragem_s : float
        Período de amostragem em segundos.

    maximo_avaliacoes : int, default=500
        Número máximo de avaliações da função de resíduos.

    Returns
    -------
    ResultadoOtimizacao2R2C
        Resultado completo da identificação.
    """

    vetor_inicial_log10 = validar_ponto_inicial(
        vetor_inicial_log10,
        limites,
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
        limites_inferiores_log10,
        limites_superiores_log10,
    ) = obter_limites_log10(
        limites
    )

    # --------------------------------------------------------
    # Avaliação do modelo antes da otimização
    # --------------------------------------------------------

    parametros_iniciais = (
        converter_log10_para_parametros(
            vetor_inicial_log10
        )
    )

    temperatura_inicial = (
        simular_para_identificacao(
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

    metricas_iniciais = calcular_metricas_ajuste(
        saida_medida,
        temperatura_inicial,
    )

    # --------------------------------------------------------
    # Execução do algoritmo de mínimos quadrados
    # --------------------------------------------------------

    resultado_scipy = least_squares(
        fun=calcular_residuos_identificacao,
        x0=vetor_inicial_log10,
        bounds=(
            limites_inferiores_log10,
            limites_superiores_log10,
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

    # --------------------------------------------------------
    # Simulação com os parâmetros otimizados
    # --------------------------------------------------------

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

    metricas_otimizadas = calcular_metricas_ajuste(
        saida_medida,
        temperatura_otimizada,
    )

    # --------------------------------------------------------
    # Avaliação da melhoria
    # --------------------------------------------------------

    melhoria_rmse_percentual = (
        calcular_melhoria_rmse_percentual(
            metricas_iniciais.rmse,
            metricas_otimizadas.rmse,
        )
    )

    parametros_proximos_limites = (
        identificar_parametros_proximos_limites(
            vetor_otimizado_log10,
            limites,
        )
    )

    return ResultadoOtimizacao2R2C(
        vetor_inicial_log10=(
            vetor_inicial_log10.copy()
        ),
        vetor_otimizado_log10=(
            vetor_otimizado_log10.copy()
        ),
        parametros_iniciais=(
            parametros_iniciais
        ),
        parametros_otimizados=(
            parametros_otimizados
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
        parametros_proximos_limites=(
            parametros_proximos_limites
        ),
    )


# ============================================================
# Validação do resultado da otimização
# ============================================================


def validar_resultado_otimizacao(
    resultado: ResultadoOtimizacao2R2C,
) -> None:
    """
    Valida estruturalmente o resultado de uma otimização.

    Parameters
    ----------
    resultado : ResultadoOtimizacao2R2C
        Resultado a validar.
    """

    if not isinstance(
        resultado,
        ResultadoOtimizacao2R2C,
    ):
        raise TypeError(
            "O resultado deve ser uma instância de "
            "ResultadoOtimizacao2R2C."
        )

    validar_vetor_log10(
        resultado.vetor_inicial_log10
    )

    validar_vetor_log10(
        resultado.vetor_otimizado_log10
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
            resultado.temperatura_inicial
        )
    ):
        raise ValueError(
            "A temperatura inicial contém valores "
            "não finitos."
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
# Apresentação do resultado
# ============================================================


def mostrar_resultado_otimizacao(
    resultado: ResultadoOtimizacao2R2C,
) -> None:
    """
    Apresenta um resumo completo da otimização.

    Parameters
    ----------
    resultado : ResultadoOtimizacao2R2C
        Resultado da otimização.
    """

    validar_resultado_otimizacao(
        resultado
    )

    inicial = resultado.parametros_iniciais
    final = resultado.parametros_otimizados

    metricas_i = resultado.metricas_iniciais
    metricas_f = resultado.metricas_otimizadas

    print(
        "========== OTIMIZAÇÃO DO MODELO 2R2C ==========\n"
    )

    print(
        f"Convergência: {resultado.sucesso}"
    )

    print(
        f"Estado do otimizador: {resultado.estado}"
    )

    print(
        f"Número de avaliações: "
        f"{resultado.numero_avaliacoes}"
    )

    print(
        f"Mensagem: {resultado.mensagem}"
    )

    print(
        "\nParâmetros iniciais:"
    )

    print(
        f"R_ext  = {inicial.R_ext:.6e} K/W"
    )

    print(
        f"R_m    = {inicial.R_m:.6e} K/W"
    )

    print(
        f"C_int  = {inicial.C_int:.6e} J/K"
    )

    print(
        f"C_m    = {inicial.C_m:.6e} J/K"
    )

    print(
        f"K_HVAC = {inicial.K_HVAC:.6f}"
    )

    print(
        "\nParâmetros otimizados:"
    )

    print(
        f"R_ext  = {final.R_ext:.6e} K/W"
    )

    print(
        f"R_m    = {final.R_m:.6e} K/W"
    )

    print(
        f"C_int  = {final.C_int:.6e} J/K"
    )

    print(
        f"C_m    = {final.C_m:.6e} J/K"
    )

    print(
        f"K_HVAC = {final.K_HVAC:.6f}"
    )

    print(
        "\nMétricas iniciais:"
    )

    print(
        f"RMSE = {metricas_i.rmse:.6f} °C"
    )

    print(
        f"MAE = {metricas_i.mae:.6f} °C"
    )

    print(
        f"Erro médio = "
        f"{metricas_i.erro_medio:.6f} °C"
    )

    print(
        f"R² = {metricas_i.r2:.6f}"
    )

    print(
        "\nMétricas otimizadas:"
    )

    print(
        f"RMSE = {metricas_f.rmse:.6f} °C"
    )

    print(
        f"MAE = {metricas_f.mae:.6f} °C"
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
        f"R² = {metricas_f.r2:.6f}"
    )

    print(
        "\nMelhoria do RMSE:"
    )

    print(
        f"{resultado.melhoria_rmse_percentual:.2f} %"
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
        "\n=============================================="
    )