"""
============================================================
Módulo: Previsao_2R2C.py

Descrição:
    Funções e estruturas destinadas à avaliação da capacidade
    preditiva do modelo térmico 2R2C em diferentes horizontes
    temporais.

    A metodologia utilizada corresponde a uma previsão
    condicional a partir de cada instante disponível.

    Em cada instante de origem:

        - T_int é atualizado com a temperatura interior medida;
        - T_m é mantido como estado térmico estimado pelo modelo;
        - o modelo evolui livremente ao longo do horizonte;
        - são utilizadas as entradas futuras experimentais
          T_ext e P_el;
        - a temperatura prevista no final do horizonte é
          comparada com a temperatura efetivamente medida.

    Esta abordagem permite avaliar a degradação da capacidade
    preditiva do modelo com o aumento do horizonte temporal.

    A análise constitui uma etapa preparatória para a escolha
    do horizonte de predição do controlador MPC.

Autor:
    Paulo Ribeiro

Projeto:
    Plataforma de Processamento de Dados Experimentais
    Dissertação de Mestrado
============================================================
"""

from dataclasses import dataclass

import numpy as np


# ============================================================
# Estruturas de dados
# ============================================================


@dataclass(frozen=True)
class MetricasPrevisao2R2C:
    """
    Métricas de desempenho para um determinado horizonte.

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
    """

    rmse: float
    mae: float
    erro_medio: float
    erro_maximo_absoluto: float
    r2: float


@dataclass(frozen=True)
class ResultadoHorizontePrevisao2R2C:
    """
    Resultado da previsão para um horizonte específico.

    Atributos
    ---------
    horizonte_amostras : int
        Horizonte expresso em número de amostras.

    horizonte_min : float
        Horizonte expresso em minutos.

    indices_origem : np.ndarray
        Índices dos instantes de início das previsões.

    indices_alvo : np.ndarray
        Índices dos instantes previstos.

    temperatura_prevista : np.ndarray
        Temperaturas previstas no final do horizonte.

    temperatura_medida : np.ndarray
        Temperaturas medidas nos instantes alvo.

    residuos : np.ndarray
        Resíduos:

            e = T_medida - T_prevista

    metricas : MetricasPrevisao2R2C
        Métricas de desempenho.

    numero_previsoes : int
        Número total de previsões válidas.
    """

    horizonte_amostras: int
    horizonte_min: float

    indices_origem: np.ndarray
    indices_alvo: np.ndarray

    temperatura_prevista: np.ndarray
    temperatura_medida: np.ndarray

    residuos: np.ndarray

    metricas: MetricasPrevisao2R2C

    numero_previsoes: int


@dataclass(frozen=True)
class ResultadoAnalisePreditiva2R2C:
    """
    Resultado completo da análise em múltiplos horizontes.

    Atributos
    ---------
    resultados : tuple[ResultadoHorizontePrevisao2R2C, ...]
        Resultados individuais de cada horizonte.

    horizontes_amostras : np.ndarray
        Horizontes analisados em amostras.

    horizontes_min : np.ndarray
        Horizontes analisados em minutos.

    numero_amostras : int
        Número total de amostras disponíveis.

    periodo_amostragem_s : float
        Período de amostragem utilizado.
    """

    resultados: tuple[ResultadoHorizontePrevisao2R2C, ...]

    horizontes_amostras: np.ndarray
    horizontes_min: np.ndarray

    numero_amostras: int

    periodo_amostragem_s: float


# ============================================================
# Validação das matrizes discretas
# ============================================================


def validar_matrizes_discretas_previsao(
    matrizes,
) -> None:
    """
    Valida as matrizes discretas necessárias à previsão.

    A função utiliza os atributos:

        Ad
        Bd
        Cd
        Dd

    permitindo reutilizar diretamente a estrutura discreta
    definida no módulo Modelo_2R2C.py.

    Parameters
    ----------
    matrizes
        Estrutura contendo as matrizes discretas.
    """

    atributos_obrigatorios = (
        "Ad",
        "Bd",
        "Cd",
        "Dd",
    )

    for atributo in atributos_obrigatorios:

        if not hasattr(
            matrizes,
            atributo,
        ):
            raise TypeError(
                "A estrutura de matrizes discretas não "
                f"contém o atributo obrigatório '{atributo}'."
            )

    Ad = np.asarray(
        matrizes.Ad,
        dtype=float,
    )

    Bd = np.asarray(
        matrizes.Bd,
        dtype=float,
    )

    Cd = np.asarray(
        matrizes.Cd,
        dtype=float,
    )

    Dd = np.asarray(
        matrizes.Dd,
        dtype=float,
    )

    if Ad.shape != (2, 2):
        raise ValueError(
            "A matriz Ad deve possuir dimensão (2, 2)."
        )

    if Bd.shape != (2, 2):
        raise ValueError(
            "A matriz Bd deve possuir dimensão (2, 2)."
        )

    if Cd.shape != (1, 2):
        raise ValueError(
            "A matriz Cd deve possuir dimensão (1, 2)."
        )

    if Dd.shape != (1, 2):
        raise ValueError(
            "A matriz Dd deve possuir dimensão (1, 2)."
        )

    for nome, matriz in {
        "Ad": Ad,
        "Bd": Bd,
        "Cd": Cd,
        "Dd": Dd,
    }.items():

        if not np.all(
            np.isfinite(
                matriz
            )
        ):
            raise ValueError(
                f"A matriz {nome} contém valores não finitos."
            )


# ============================================================
# Validação do período de amostragem
# ============================================================


def validar_periodo_amostragem_previsao(
    periodo_amostragem_s: float,
) -> float:
    """
    Valida o período de amostragem utilizado na previsão.
    """

    if not isinstance(
        periodo_amostragem_s,
        (int, float, np.integer, np.floating),
    ):
        raise TypeError(
            "O período de amostragem deve ser numérico."
        )

    periodo_amostragem_s = float(
        periodo_amostragem_s
    )

    if not np.isfinite(
        periodo_amostragem_s
    ):
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
# Validação dos dados de previsão
# ============================================================


def validar_dados_previsao(
    entradas: np.ndarray,
    saida_medida: np.ndarray,
    estados_modelo: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Valida os dados utilizados na análise preditiva.

    Parameters
    ----------
    entradas : np.ndarray
        Matriz de entradas com dimensão (N, 2).

    saida_medida : np.ndarray
        Temperatura interior medida com dimensão (N,).

    estados_modelo : np.ndarray
        Estados obtidos através da propagação do modelo,
        com dimensão (N + 1, 2).

    Returns
    -------
    tuple
        Entradas, saída e estados validados.
    """

    try:

        entradas = np.asarray(
            entradas,
            dtype=float,
        )

        saida_medida = np.asarray(
            saida_medida,
            dtype=float,
        )

        estados_modelo = np.asarray(
            estados_modelo,
            dtype=float,
        )

    except (TypeError, ValueError) as erro:

        raise TypeError(
            "Os dados de previsão devem conter apenas "
            "valores numéricos."
        ) from erro

    if entradas.ndim != 2:
        raise ValueError(
            "As entradas devem ser bidimensionais."
        )

    if entradas.shape[1] != 2:
        raise ValueError(
            "As entradas devem possuir duas colunas: "
            "[T_ext, P_el]."
        )

    numero_amostras = entradas.shape[0]

    if saida_medida.shape != (
        numero_amostras,
    ):
        raise ValueError(
            "A saída medida deve possuir dimensão "
            f"({numero_amostras},)."
        )

    if estados_modelo.shape != (
        numero_amostras + 1,
        2,
    ):
        raise ValueError(
            "Os estados do modelo devem possuir dimensão "
            f"({numero_amostras + 1}, 2)."
        )

    if not np.all(
        np.isfinite(
            entradas
        )
    ):
        raise ValueError(
            "As entradas contêm valores não finitos."
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
            estados_modelo
        )
    ):
        raise ValueError(
            "Os estados do modelo contêm valores não finitos."
        )

    return (
        entradas,
        saida_medida,
        estados_modelo,
    )


# ============================================================
# Construção dos estados de origem
# ============================================================


def construir_estados_origem_previsao(
    estados_modelo: np.ndarray,
    saida_medida: np.ndarray,
) -> np.ndarray:
    """
    Constrói os estados utilizados no início de cada previsão.

    A temperatura interior do estado é atualizada com a
    medição disponível no respetivo instante:

        T_int_estado(k) = T_int_medida(k)

    A temperatura da massa térmica permanece igual à
    estimativa fornecida pela trajetória do modelo:

        T_m_estado(k) = T_m_modelo(k)

    Esta abordagem corresponde a uma correção parcial do
    estado e não constitui um observador de estados formal.

    Parameters
    ----------
    estados_modelo : np.ndarray
        Trajetória dos estados do modelo com dimensão
        (N + 1, 2).

    saida_medida : np.ndarray
        Temperatura interior medida com dimensão (N,).

    Returns
    -------
    np.ndarray
        Estados de origem com dimensão (N, 2).
    """

    estados_modelo = np.asarray(
        estados_modelo,
        dtype=float,
    )

    saida_medida = np.asarray(
        saida_medida,
        dtype=float,
    )

    numero_amostras = saida_medida.size

    if estados_modelo.shape != (
        numero_amostras + 1,
        2,
    ):
        raise ValueError(
            "A dimensão dos estados não é compatível "
            "com a saída medida."
        )

    estados_origem = (
        estados_modelo[:-1]
        .copy()
    )

    estados_origem[:, 0] = (
        saida_medida
    )

    if not np.all(
        np.isfinite(
            estados_origem
        )
    ):
        raise ValueError(
            "Os estados de origem contêm valores não finitos."
        )

    return estados_origem


# ============================================================
# Cálculo das métricas
# ============================================================


def calcular_metricas_previsao(
    temperatura_medida: np.ndarray,
    temperatura_prevista: np.ndarray,
) -> MetricasPrevisao2R2C:
    """
    Calcula as métricas de desempenho da previsão.
    """

    temperatura_medida = np.asarray(
        temperatura_medida,
        dtype=float,
    )

    temperatura_prevista = np.asarray(
        temperatura_prevista,
        dtype=float,
    )

    if (
        temperatura_medida.shape
        != temperatura_prevista.shape
    ):
        raise ValueError(
            "As temperaturas medida e prevista devem "
            "possuir a mesma dimensão."
        )

    if temperatura_medida.ndim != 1:
        raise ValueError(
            "As sequências de temperatura devem ser "
            "unidimensionais."
        )

    if temperatura_medida.size == 0:
        raise ValueError(
            "Não existem previsões para calcular métricas."
        )

    if not np.all(
        np.isfinite(
            temperatura_medida
        )
    ):
        raise ValueError(
            "A temperatura medida contém valores não finitos."
        )

    if not np.all(
        np.isfinite(
            temperatura_prevista
        )
    ):
        raise ValueError(
            "A temperatura prevista contém valores não finitos."
        )

    residuos = (
        temperatura_medida
        - temperatura_prevista
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
                temperatura_medida
                - np.mean(
                    temperatura_medida
                )
            ) ** 2
        )
    )

    if soma_quadrados_total <= 0:

        r2 = np.nan

    else:

        r2 = float(
            1.0
            - soma_quadrados_residuos
            / soma_quadrados_total
        )

    return MetricasPrevisao2R2C(
        rmse=rmse,
        mae=mae,
        erro_medio=erro_medio,
        erro_maximo_absoluto=erro_maximo_absoluto,
        r2=r2,
    )


# ============================================================
# Previsão para um horizonte
# ============================================================


def prever_horizonte_2r2c(
    matrizes,
    entradas: np.ndarray,
    saida_medida: np.ndarray,
    estados_origem: np.ndarray,
    horizonte_amostras: int,
    periodo_amostragem_s: float,
) -> ResultadoHorizontePrevisao2R2C:
    """
    Executa previsões de temperatura para um horizonte fixo.

    Para cada instante k:

        1. utiliza-se o estado corrigido x(k);
        2. propagam-se as entradas de k até k + Np - 1;
        3. calcula-se T_int prevista em k + Np;
        4. compara-se com T_int medida em k + Np.

    Parameters
    ----------
    matrizes
        Matrizes discretas do modelo 2R2C.

    entradas : np.ndarray
        Entradas [T_ext, P_el].

    saida_medida : np.ndarray
        Temperatura interior medida.

    estados_origem : np.ndarray
        Estados corrigidos de início das previsões, com
        dimensão (N, 2).

    horizonte_amostras : int
        Horizonte em número de amostras.

    periodo_amostragem_s : float
        Período de amostragem em segundos.

    Returns
    -------
    ResultadoHorizontePrevisao2R2C
        Resultado completo do horizonte.
    """

    validar_matrizes_discretas_previsao(
        matrizes
    )

    periodo_amostragem_s = (
        validar_periodo_amostragem_previsao(
            periodo_amostragem_s
        )
    )

    entradas = np.asarray(
        entradas,
        dtype=float,
    )

    saida_medida = np.asarray(
        saida_medida,
        dtype=float,
    )

    estados_origem = np.asarray(
        estados_origem,
        dtype=float,
    )

    numero_amostras = (
        entradas.shape[0]
    )

    if estados_origem.shape != (
        numero_amostras,
        2,
    ):
        raise ValueError(
            "Os estados de origem devem possuir dimensão "
            f"({numero_amostras}, 2)."
        )

    if saida_medida.shape != (
        numero_amostras,
    ):
        raise ValueError(
            "A saída medida apresenta dimensão incompatível."
        )

    if not isinstance(
        horizonte_amostras,
        (int, np.integer),
    ):
        raise TypeError(
            "O horizonte deve ser um número inteiro "
            "de amostras."
        )

    horizonte_amostras = int(
        horizonte_amostras
    )

    if horizonte_amostras <= 0:
        raise ValueError(
            "O horizonte deve ser estritamente positivo."
        )

    if horizonte_amostras >= numero_amostras:
        raise ValueError(
            "O horizonte deve ser inferior ao número "
            "de amostras disponíveis."
        )

    Ad = np.asarray(
        matrizes.Ad,
        dtype=float,
    )

    Bd = np.asarray(
        matrizes.Bd,
        dtype=float,
    )

    Cd = np.asarray(
        matrizes.Cd,
        dtype=float,
    )

    Dd = np.asarray(
        matrizes.Dd,
        dtype=float,
    )

    numero_previsoes = (
        numero_amostras
        - horizonte_amostras
    )

    indices_origem = np.arange(
        numero_previsoes,
        dtype=int,
    )

    indices_alvo = (
        indices_origem
        + horizonte_amostras
    )

    temperatura_prevista = np.empty(
        numero_previsoes,
        dtype=float,
    )

    # --------------------------------------------------------
    # Execução das previsões
    # --------------------------------------------------------

    for indice_previsao, indice_origem in enumerate(
        indices_origem
    ):

        estado = (
            estados_origem[
                indice_origem
            ]
            .copy()
        )

        indice_final_propagacao = (
            indice_origem
            + horizonte_amostras
        )

        for indice in range(
            indice_origem,
            indice_final_propagacao,
        ):

            estado = (
                Ad @ estado
                + Bd @ entradas[indice]
            )

        indice_alvo = (
            indice_final_propagacao
        )

        saida_prevista = (
            Cd @ estado
            + Dd @ entradas[indice_alvo]
        )

        temperatura_prevista[
            indice_previsao
        ] = float(
            saida_prevista[0]
        )

    temperatura_medida_alvo = (
        saida_medida[
            indices_alvo
        ]
        .copy()
    )

    residuos = (
        temperatura_medida_alvo
        - temperatura_prevista
    )

    metricas = calcular_metricas_previsao(
        temperatura_medida=(
            temperatura_medida_alvo
        ),
        temperatura_prevista=(
            temperatura_prevista
        ),
    )

    horizonte_min = float(
        horizonte_amostras
        * periodo_amostragem_s
        / 60.0
    )

    return ResultadoHorizontePrevisao2R2C(
        horizonte_amostras=(
            horizonte_amostras
        ),
        horizonte_min=(
            horizonte_min
        ),
        indices_origem=(
            indices_origem
        ),
        indices_alvo=(
            indices_alvo
        ),
        temperatura_prevista=(
            temperatura_prevista
        ),
        temperatura_medida=(
            temperatura_medida_alvo
        ),
        residuos=(
            residuos
        ),
        metricas=(
            metricas
        ),
        numero_previsoes=(
            numero_previsoes
        ),
    )


# ============================================================
# Análise em múltiplos horizontes
# ============================================================


def analisar_horizontes_previsao_2r2c(
    matrizes,
    entradas: np.ndarray,
    saida_medida: np.ndarray,
    estados_modelo: np.ndarray,
    horizontes_amostras: np.ndarray,
    periodo_amostragem_s: float,
) -> ResultadoAnalisePreditiva2R2C:
    """
    Avalia a capacidade preditiva do modelo em vários
    horizontes temporais.

    Parameters
    ----------
    matrizes
        Matrizes discretas do modelo.

    entradas : np.ndarray
        Entradas experimentais.

    saida_medida : np.ndarray
        Temperatura interior medida.

    estados_modelo : np.ndarray
        Trajetória de estados obtida pelo modelo.

    horizontes_amostras : np.ndarray
        Horizontes a analisar.

    periodo_amostragem_s : float
        Período de amostragem.

    Returns
    -------
    ResultadoAnalisePreditiva2R2C
        Resultado completo da análise.
    """

    validar_matrizes_discretas_previsao(
        matrizes
    )

    (
        entradas,
        saida_medida,
        estados_modelo,
    ) = validar_dados_previsao(
        entradas=entradas,
        saida_medida=saida_medida,
        estados_modelo=estados_modelo,
    )

    periodo_amostragem_s = (
        validar_periodo_amostragem_previsao(
            periodo_amostragem_s
        )
    )

    try:

        horizontes_amostras = np.asarray(
            horizontes_amostras,
            dtype=int,
        )

    except (TypeError, ValueError) as erro:

        raise TypeError(
            "Os horizontes devem conter valores inteiros."
        ) from erro

    if horizontes_amostras.ndim != 1:
        raise ValueError(
            "Os horizontes devem ser unidimensionais."
        )

    if horizontes_amostras.size == 0:
        raise ValueError(
            "Deve ser definido pelo menos um horizonte."
        )

    if np.any(
        horizontes_amostras <= 0
    ):
        raise ValueError(
            "Todos os horizontes devem ser positivos."
        )

    if np.any(
        horizontes_amostras
        >= entradas.shape[0]
    ):
        raise ValueError(
            "Todos os horizontes devem ser inferiores "
            "ao número de amostras disponíveis."
        )

    if np.unique(
        horizontes_amostras
    ).size != horizontes_amostras.size:

        raise ValueError(
            "Os horizontes não podem conter duplicados."
        )

    estados_origem = (
        construir_estados_origem_previsao(
            estados_modelo=estados_modelo,
            saida_medida=saida_medida,
        )
    )

    resultados = []

    for horizonte in (
        horizontes_amostras
    ):

        resultado = prever_horizonte_2r2c(
            matrizes=matrizes,
            entradas=entradas,
            saida_medida=saida_medida,
            estados_origem=estados_origem,
            horizonte_amostras=int(
                horizonte
            ),
            periodo_amostragem_s=(
                periodo_amostragem_s
            ),
        )

        resultados.append(
            resultado
        )

    horizontes_min = (
        horizontes_amostras.astype(float)
        * periodo_amostragem_s
        / 60.0
    )

    return ResultadoAnalisePreditiva2R2C(
        resultados=tuple(
            resultados
        ),
        horizontes_amostras=(
            horizontes_amostras.copy()
        ),
        horizontes_min=(
            horizontes_min
        ),
        numero_amostras=int(
            entradas.shape[0]
        ),
        periodo_amostragem_s=(
            periodo_amostragem_s
        ),
    )


# ============================================================
# Validação do resultado completo
# ============================================================


def validar_resultado_analise_preditiva(
    resultado: ResultadoAnalisePreditiva2R2C,
) -> None:
    """
    Valida estruturalmente o resultado da análise preditiva.
    """

    if not isinstance(
        resultado,
        ResultadoAnalisePreditiva2R2C,
    ):
        raise TypeError(
            "O resultado deve ser uma instância de "
            "ResultadoAnalisePreditiva2R2C."
        )

    if (
        len(resultado.resultados)
        != resultado.horizontes_amostras.size
    ):
        raise ValueError(
            "O número de resultados não corresponde "
            "ao número de horizontes."
        )

    for resultado_horizonte in (
        resultado.resultados
    ):

        if (
            resultado_horizonte.numero_previsoes
            <= 0
        ):
            raise ValueError(
                "Foi encontrado um horizonte sem "
                "previsões válidas."
            )

        if not np.all(
            np.isfinite(
                resultado_horizonte
                .temperatura_prevista
            )
        ):
            raise ValueError(
                "Foram encontradas previsões não finitas."
            )

        if not np.all(
            np.isfinite(
                resultado_horizonte
                .residuos
            )
        ):
            raise ValueError(
                "Foram encontrados resíduos não finitos."
            )


# ============================================================
# Apresentação do resumo
# ============================================================


def mostrar_resumo_analise_preditiva(
    resultado: ResultadoAnalisePreditiva2R2C,
) -> None:
    """
    Apresenta uma tabela textual com as métricas obtidas em
    todos os horizontes analisados.
    """

    validar_resultado_analise_preditiva(
        resultado
    )

    print(
        "========== CAPACIDADE PREDITIVA DO MODELO 2R2C ==========\n"
    )

    print(
        "Horizonte | Previsões | RMSE [°C] | MAE [°C] | "
        "Erro médio [°C] | Erro máx. [°C] | R²"
    )

    print(
        "-" * 92
    )

    for resultado_horizonte in (
        resultado.resultados
    ):

        metricas = (
            resultado_horizonte.metricas
        )

        print(
            f"{resultado_horizonte.horizonte_min:8.0f} min | "
            f"{resultado_horizonte.numero_previsoes:9d} | "
            f"{metricas.rmse:9.6f} | "
            f"{metricas.mae:8.6f} | "
            f"{metricas.erro_medio:15.6f} | "
            f"{metricas.erro_maximo_absoluto:14.6f} | "
            f"{metricas.r2:8.6f}"
        )

    print(
        "\n========================================================="
    )