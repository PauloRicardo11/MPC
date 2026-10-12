"""
============================================================
Módulo: Sincronizacao.py

Descrição:
    Funções destinadas à preparação, validação e sincronização
    temporal dos dados experimentais utilizados no projeto.

    O módulo permite:

    - construir referências temporais absolutas;
    - validar sequências temporais absolutas;
    - validar sequências de tempo relativo, em segundos;
    - caracterizar os intervalos de amostragem;
    - atribuir uma referência temporal absoluta a dados cuja
      aquisição foi registada através de tempo relativo;
    - sincronizar medições provenientes de diferentes sistemas
      de aquisição através de interpolação temporal.

    Todas as operações são realizadas sobre cópias dos
    DataFrames recebidos, preservando os dados de origem.

Autor:
    Paulo Ribeiro

Projeto:
    Plataforma de Processamento de Dados Experimentais
============================================================
"""

from collections.abc import Sequence
from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class ResultadoValidacaoTemporal:
    """
    Resultado estruturado da validação temporal absoluta.

    Attributes
    ----------
    valido:
        Indica se a sequência temporal cumpre os critérios
        definidos.

    numero_registos:
        Número total de instantes temporais analisados.

    primeiro_instante:
        Primeiro instante temporal do conjunto.

    ultimo_instante:
        Último instante temporal do conjunto.

    monotonicamente_crescente:
        Indica se os instantes se encontram ordenados.

    numero_instantes_duplicados:
        Número de instantes temporais repetidos.

    intervalo_mediano_s:
        Intervalo mediano entre medições, em segundos.

    intervalo_minimo_s:
        Menor intervalo entre medições, em segundos.

    intervalo_maximo_s:
        Maior intervalo entre medições, em segundos.

    numero_intervalos_irregulares:
        Número de intervalos fora da tolerância definida.

    intervalo_esperado_s:
        Intervalo de amostragem esperado, em segundos.

    tolerancia_s:
        Tolerância admitida relativamente ao intervalo esperado,
        em segundos.
    """

    valido: bool
    numero_registos: int
    primeiro_instante: pd.Timestamp
    ultimo_instante: pd.Timestamp
    monotonicamente_crescente: bool
    numero_instantes_duplicados: int
    intervalo_mediano_s: float
    intervalo_minimo_s: float
    intervalo_maximo_s: float
    numero_intervalos_irregulares: int
    intervalo_esperado_s: float
    tolerancia_s: float


@dataclass(frozen=True)
class ResultadoValidacaoTempoRelativo:
    """
    Resultado estruturado da validação de uma sequência temporal
    relativa expressa em segundos.

    Attributes
    ----------
    valido:
        Indica se a sequência temporal cumpre os critérios
        definidos.

    numero_registos:
        Número total de registos analisados.

    primeiro_tempo_s:
        Primeiro valor temporal, em segundos.

    ultimo_tempo_s:
        Último valor temporal, em segundos.

    duracao_total_s:
        Duração entre o primeiro e o último registo.

    monotonicamente_crescente:
        Indica se a sequência temporal é estritamente crescente.

    numero_instantes_duplicados:
        Número de valores temporais repetidos.

    intervalo_mediano_s:
        Intervalo mediano entre medições.

    intervalo_medio_s:
        Intervalo médio entre medições.

    intervalo_minimo_s:
        Menor intervalo entre medições.

    intervalo_maximo_s:
        Maior intervalo entre medições.

    numero_intervalos_irregulares:
        Número de intervalos fora da tolerância admitida.

    intervalo_esperado_s:
        Intervalo de amostragem esperado.

    tolerancia_s:
        Desvio máximo admitido relativamente ao intervalo esperado.
    """

    valido: bool
    numero_registos: int
    primeiro_tempo_s: float
    ultimo_tempo_s: float
    duracao_total_s: float
    monotonicamente_crescente: bool
    numero_instantes_duplicados: int
    intervalo_mediano_s: float
    intervalo_medio_s: float
    intervalo_minimo_s: float
    intervalo_maximo_s: float
    numero_intervalos_irregulares: int
    intervalo_esperado_s: float
    tolerancia_s: float


def verificar_dataframe(
    dataframe: pd.DataFrame,
) -> None:
    """
    Verifica se o objeto fornecido é um DataFrame não vazio.

    Parameters
    ----------
    dataframe:
        Objeto a verificar.

    Raises
    ------
    TypeError
        Caso o objeto não seja um DataFrame do pandas.

    ValueError
        Caso o DataFrame esteja vazio.
    """

    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError(
            "O objeto fornecido não é um DataFrame do pandas."
        )

    if dataframe.empty:
        raise ValueError(
            "O DataFrame não contém dados."
        )


def criar_coluna_data_hora(
    dataframe: pd.DataFrame,
    coluna_data: str,
    coluna_hora: str,
    nome_coluna: str = "DataHora",
    formato: str = "%Y/%m/%d %H:%M:%S",
) -> pd.DataFrame:
    """
    Combina colunas de data e hora numa referência temporal única.

    Parameters
    ----------
    dataframe:
        DataFrame a processar.

    coluna_data:
        Nome da coluna que contém a data.

    coluna_hora:
        Nome da coluna que contém a hora.

    nome_coluna:
        Nome da nova coluna temporal.

    formato:
        Formato utilizado na interpretação da data e hora.

    Returns
    -------
    pandas.DataFrame
        Novo DataFrame contendo a coluna temporal criada.
    """

    verificar_dataframe(dataframe)

    parametros_texto = {
        "coluna_data": coluna_data,
        "coluna_hora": coluna_hora,
        "nome_coluna": nome_coluna,
        "formato": formato,
    }

    for nome_parametro, valor in parametros_texto.items():

        if not isinstance(valor, str):
            raise TypeError(
                f"O parâmetro '{nome_parametro}' deve ser uma string."
            )

        if not valor.strip():
            raise ValueError(
                f"O parâmetro '{nome_parametro}' não pode ser vazio."
            )

    colunas_em_falta = [
        coluna
        for coluna in (coluna_data, coluna_hora)
        if coluna not in dataframe.columns
    ]

    if colunas_em_falta:
        raise ValueError(
            "As seguintes colunas não existem no DataFrame: "
            f"{colunas_em_falta}"
        )

    if (
        nome_coluna in dataframe.columns
        and nome_coluna not in (coluna_data, coluna_hora)
    ):
        raise ValueError(
            f"A coluna '{nome_coluna}' já existe no DataFrame."
        )

    dataframe_temporal = dataframe.copy()

    valores_data_hora = (
        dataframe_temporal[coluna_data]
        .astype(str)
        .str.strip()
        + " "
        + dataframe_temporal[coluna_hora]
        .astype(str)
        .str.strip()
    )

    try:
        coluna_temporal = pd.to_datetime(
            valores_data_hora,
            format=formato,
            errors="raise",
        )

    except (TypeError, ValueError) as erro:
        raise ValueError(
            "Não foi possível converter as colunas "
            f"'{coluna_data}' e '{coluna_hora}' para datetime."
        ) from erro

    dataframe_temporal.insert(
        0,
        nome_coluna,
        coluna_temporal,
    )

    return dataframe_temporal


def validar_sequencia_temporal(
    dataframe: pd.DataFrame,
    coluna_temporal: str,
    intervalo_esperado_s: float,
    tolerancia_s: float = 0.0,
) -> ResultadoValidacaoTemporal:
    """
    Valida a coerência de uma sequência temporal absoluta.
    """

    verificar_dataframe(dataframe)

    if not isinstance(coluna_temporal, str):
        raise TypeError(
            "O nome da coluna temporal deve ser uma string."
        )

    if not coluna_temporal.strip():
        raise ValueError(
            "O nome da coluna temporal não pode ser vazio."
        )

    if coluna_temporal not in dataframe.columns:
        raise ValueError(
            f"A coluna temporal '{coluna_temporal}' "
            "não existe no DataFrame."
        )

    if not isinstance(intervalo_esperado_s, (int, float)):
        raise TypeError(
            "O intervalo esperado deve ser um valor numérico."
        )

    if intervalo_esperado_s <= 0:
        raise ValueError(
            "O intervalo esperado deve ser superior a zero."
        )

    if not isinstance(tolerancia_s, (int, float)):
        raise TypeError(
            "A tolerância deve ser um valor numérico."
        )

    if tolerancia_s < 0:
        raise ValueError(
            "A tolerância não pode ser negativa."
        )

    if not pd.api.types.is_datetime64_any_dtype(
        dataframe[coluna_temporal]
    ):
        raise TypeError(
            f"A coluna '{coluna_temporal}' não possui tipo datetime."
        )

    if len(dataframe) < 2:
        raise ValueError(
            "São necessários pelo menos dois registos para "
            "validar a sequência temporal."
        )

    serie_temporal = dataframe[coluna_temporal]

    monotonicamente_crescente = bool(
        serie_temporal.is_monotonic_increasing
    )

    numero_instantes_duplicados = int(
        serie_temporal.duplicated().sum()
    )

    intervalos_s = (
        serie_temporal
        .diff()
        .dropna()
        .dt.total_seconds()
    )

    desvios_s = (
        intervalos_s
        - float(intervalo_esperado_s)
    ).abs()

    numero_intervalos_irregulares = int(
        (desvios_s > float(tolerancia_s)).sum()
    )

    valido = (
        monotonicamente_crescente
        and numero_instantes_duplicados == 0
        and numero_intervalos_irregulares == 0
    )

    return ResultadoValidacaoTemporal(
        valido=valido,
        numero_registos=int(len(dataframe)),
        primeiro_instante=pd.Timestamp(
            serie_temporal.iloc[0]
        ),
        ultimo_instante=pd.Timestamp(
            serie_temporal.iloc[-1]
        ),
        monotonicamente_crescente=monotonicamente_crescente,
        numero_instantes_duplicados=numero_instantes_duplicados,
        intervalo_mediano_s=float(
            intervalos_s.median()
        ),
        intervalo_minimo_s=float(
            intervalos_s.min()
        ),
        intervalo_maximo_s=float(
            intervalos_s.max()
        ),
        numero_intervalos_irregulares=(
            numero_intervalos_irregulares
        ),
        intervalo_esperado_s=float(
            intervalo_esperado_s
        ),
        tolerancia_s=float(
            tolerancia_s
        ),
    )


def validar_tempo_relativo(
    dataframe: pd.DataFrame,
    coluna_temporal: str,
    intervalo_esperado_s: float,
    tolerancia_s: float,
) -> ResultadoValidacaoTempoRelativo:
    """
    Valida uma sequência temporal relativa expressa em segundos.
    """

    verificar_dataframe(dataframe)

    if not isinstance(coluna_temporal, str):
        raise TypeError(
            "O nome da coluna temporal deve ser uma string."
        )

    if not coluna_temporal.strip():
        raise ValueError(
            "O nome da coluna temporal não pode ser vazio."
        )

    if coluna_temporal not in dataframe.columns:
        raise ValueError(
            f"A coluna temporal '{coluna_temporal}' "
            "não existe no DataFrame."
        )

    if not isinstance(intervalo_esperado_s, (int, float)):
        raise TypeError(
            "O intervalo esperado deve ser um valor numérico."
        )

    if intervalo_esperado_s <= 0:
        raise ValueError(
            "O intervalo esperado deve ser superior a zero."
        )

    if not isinstance(tolerancia_s, (int, float)):
        raise TypeError(
            "A tolerância deve ser um valor numérico."
        )

    if tolerancia_s < 0:
        raise ValueError(
            "A tolerância não pode ser negativa."
        )

    if len(dataframe) < 2:
        raise ValueError(
            "São necessários pelo menos dois registos para "
            "validar a sequência temporal."
        )

    serie_temporal = dataframe[coluna_temporal]

    if serie_temporal.isna().any():
        raise ValueError(
            f"A coluna '{coluna_temporal}' contém valores em falta."
        )

    if not pd.api.types.is_numeric_dtype(
        serie_temporal
    ):
        raise TypeError(
            f"A coluna '{coluna_temporal}' não possui tipo numérico."
        )

    serie_temporal = serie_temporal.astype(float)

    intervalos_s = (
        serie_temporal
        .diff()
        .dropna()
    )

    monotonicamente_crescente = bool(
        (intervalos_s > 0).all()
    )

    numero_instantes_duplicados = int(
        serie_temporal.duplicated().sum()
    )

    desvios_s = (
        intervalos_s
        - float(intervalo_esperado_s)
    ).abs()

    numero_intervalos_irregulares = int(
        (desvios_s > float(tolerancia_s)).sum()
    )

    valido = (
        monotonicamente_crescente
        and numero_instantes_duplicados == 0
        and numero_intervalos_irregulares == 0
    )

    primeiro_tempo_s = float(
        serie_temporal.iloc[0]
    )

    ultimo_tempo_s = float(
        serie_temporal.iloc[-1]
    )

    return ResultadoValidacaoTempoRelativo(
        valido=valido,
        numero_registos=int(len(dataframe)),
        primeiro_tempo_s=primeiro_tempo_s,
        ultimo_tempo_s=ultimo_tempo_s,
        duracao_total_s=(
            ultimo_tempo_s
            - primeiro_tempo_s
        ),
        monotonicamente_crescente=(
            monotonicamente_crescente
        ),
        numero_instantes_duplicados=(
            numero_instantes_duplicados
        ),
        intervalo_mediano_s=float(
            intervalos_s.median()
        ),
        intervalo_medio_s=float(
            intervalos_s.mean()
        ),
        intervalo_minimo_s=float(
            intervalos_s.min()
        ),
        intervalo_maximo_s=float(
            intervalos_s.max()
        ),
        numero_intervalos_irregulares=(
            numero_intervalos_irregulares
        ),
        intervalo_esperado_s=float(
            intervalo_esperado_s
        ),
        tolerancia_s=float(
            tolerancia_s
        ),
    )


def criar_referencia_temporal_relativa(
    dataframe: pd.DataFrame,
    coluna_tempo_s: str,
    instante_referencia: str | pd.Timestamp,
    desfasamento_s: float = 0.0,
    nome_coluna: str = "DataHora",
) -> pd.DataFrame:
    """
    Constrói uma referência temporal absoluta para dados cuja
    aquisição foi registada através de tempo relativo.

    O primeiro registo do DataFrame é associado ao instante
    de referência acrescido do desfasamento indicado.

    Os restantes instantes são calculados preservando os
    intervalos temporais efetivamente registados no conjunto
    experimental.

    Parameters
    ----------
    dataframe:
        DataFrame que contém o tempo relativo.

    coluna_tempo_s:
        Nome da coluna com o tempo decorrido, em segundos.

    instante_referencia:
        Instante absoluto utilizado como referência.

    desfasamento_s:
        Deslocamento aplicado ao primeiro registo relativamente
        ao instante de referência.

        Um valor negativo significa que a primeira medição ocorreu
        antes do instante de referência.

        Um valor positivo significa que ocorreu depois.

    nome_coluna:
        Nome da coluna temporal criada.

    Returns
    -------
    pandas.DataFrame
        Novo DataFrame contendo a referência temporal absoluta.
    """

    verificar_dataframe(dataframe)

    if not isinstance(coluna_tempo_s, str):
        raise TypeError(
            "O nome da coluna temporal deve ser uma string."
        )

    if coluna_tempo_s not in dataframe.columns:
        raise ValueError(
            f"A coluna '{coluna_tempo_s}' não existe no DataFrame."
        )

    if not isinstance(nome_coluna, str):
        raise TypeError(
            "O nome da nova coluna deve ser uma string."
        )

    if not nome_coluna.strip():
        raise ValueError(
            "O nome da nova coluna não pode ser vazio."
        )

    if nome_coluna in dataframe.columns:
        raise ValueError(
            f"A coluna '{nome_coluna}' já existe no DataFrame."
        )

    if not isinstance(desfasamento_s, (int, float)):
        raise TypeError(
            "O desfasamento deve ser um valor numérico."
        )

    serie_tempo = dataframe[coluna_tempo_s]

    if serie_tempo.isna().any():
        raise ValueError(
            f"A coluna '{coluna_tempo_s}' contém valores em falta."
        )

    if not pd.api.types.is_numeric_dtype(
        serie_tempo
    ):
        raise TypeError(
            f"A coluna '{coluna_tempo_s}' deve ser numérica."
        )

    try:
        referencia = pd.Timestamp(
            instante_referencia
        )

    except (TypeError, ValueError) as erro:
        raise ValueError(
            "Não foi possível interpretar o instante de referência."
        ) from erro

    if pd.isna(referencia):
        raise ValueError(
            "O instante de referência não é válido."
        )

    dataframe_temporal = dataframe.copy()

    tempo_inicial_s = float(
        serie_tempo.iloc[0]
    )

    tempo_desde_primeira_medicao_s = (
        serie_tempo.astype(float)
        - tempo_inicial_s
    )

    primeiro_instante = (
        referencia
        + pd.to_timedelta(
            float(desfasamento_s),
            unit="s",
        )
    )

    coluna_data_hora = (
        primeiro_instante
        + pd.to_timedelta(
            tempo_desde_primeira_medicao_s,
            unit="s",
        )
    )

    dataframe_temporal.insert(
        0,
        nome_coluna,
        coluna_data_hora,
    )

    return dataframe_temporal


def sincronizar_por_interpolacao(
    dataframe_referencia: pd.DataFrame,
    dataframe_interpolar: pd.DataFrame,
    coluna_temporal: str,
    colunas_interpolar: Sequence[str],
) -> pd.DataFrame:
    """
    Interpola variáveis de um DataFrame nos instantes temporais
    de outro DataFrame utilizado como referência.

    Apenas são utilizados instantes do DataFrame de referência
    compreendidos entre a primeira e a última medição disponível
    no DataFrame a interpolar.

    As amostras originais imediatamente antes e depois dos
    instantes de referência são preservadas durante a interpolação,
    garantindo que não são introduzidos valores em falta devido
    ao corte prematuro das fronteiras temporais.

    Não é realizada extrapolação para instantes exteriores ao
    domínio temporal dos dados a interpolar.

    Parameters
    ----------
    dataframe_referencia:
        DataFrame que define os instantes finais da sincronização.

    dataframe_interpolar:
        DataFrame contendo as variáveis a interpolar.

    coluna_temporal:
        Nome da coluna datetime comum aos dois DataFrames.

    colunas_interpolar:
        Variáveis numéricas a interpolar.

    Returns
    -------
    pandas.DataFrame
        Novo DataFrame sincronizado.

    Raises
    ------
    TypeError
        Caso os parâmetros apresentem tipos inválidos.

    ValueError
        Caso as colunas necessárias não existam, os conjuntos
        não possuam sobreposição temporal ou permaneçam valores
        em falta após a interpolação.
    """

    verificar_dataframe(dataframe_referencia)
    verificar_dataframe(dataframe_interpolar)

    if not isinstance(coluna_temporal, str):
        raise TypeError(
            "O nome da coluna temporal deve ser uma string."
        )

    if not coluna_temporal.strip():
        raise ValueError(
            "O nome da coluna temporal não pode ser vazio."
        )

    for dataframe, nome in (
        (dataframe_referencia, "referência"),
        (dataframe_interpolar, "interpolação"),
    ):

        if coluna_temporal not in dataframe.columns:
            raise ValueError(
                f"A coluna '{coluna_temporal}' não existe "
                f"no DataFrame de {nome}."
            )

        if not pd.api.types.is_datetime64_any_dtype(
            dataframe[coluna_temporal]
        ):
            raise TypeError(
                f"A coluna '{coluna_temporal}' do DataFrame "
                f"de {nome} não possui tipo datetime."
            )

    if isinstance(colunas_interpolar, str):
        raise TypeError(
            "As colunas a interpolar devem ser fornecidas "
            "através de uma sequência de strings."
        )

    colunas_interpolar = list(
        colunas_interpolar
    )

    if not colunas_interpolar:
        raise ValueError(
            "Deve ser indicada pelo menos uma coluna "
            "para interpolação."
        )

    if not all(
        isinstance(coluna, str)
        for coluna in colunas_interpolar
    ):
        raise TypeError(
            "Todos os nomes das colunas devem ser strings."
        )

    if len(colunas_interpolar) != len(
        set(colunas_interpolar)
    ):
        raise ValueError(
            "A lista de colunas a interpolar contém "
            "nomes repetidos."
        )

    colunas_em_falta = [
        coluna
        for coluna in colunas_interpolar
        if coluna not in dataframe_interpolar.columns
    ]

    if colunas_em_falta:
        raise ValueError(
            "As seguintes colunas não existem no DataFrame "
            f"a interpolar: {colunas_em_falta}"
        )

    colunas_nao_numericas = [
        coluna
        for coluna in colunas_interpolar
        if not pd.api.types.is_numeric_dtype(
            dataframe_interpolar[coluna]
        )
    ]

    if colunas_nao_numericas:
        raise TypeError(
            "As seguintes colunas não são numéricas: "
            f"{colunas_nao_numericas}"
        )

    valores_em_falta_origem = (
        dataframe_interpolar[
            colunas_interpolar
        ]
        .isna()
        .sum()
    )

    colunas_com_faltas = (
        valores_em_falta_origem[
            valores_em_falta_origem > 0
        ]
        .index
        .tolist()
    )

    if colunas_com_faltas:
        raise ValueError(
            "As colunas a interpolar contêm valores em falta: "
            f"{colunas_com_faltas}"
        )

    referencia = (
        dataframe_referencia
        .sort_values(coluna_temporal)
        .reset_index(drop=True)
        .copy()
    )

    interpolar = (
        dataframe_interpolar[
            [coluna_temporal] + colunas_interpolar
        ]
        .sort_values(coluna_temporal)
        .reset_index(drop=True)
        .copy()
    )

    if referencia[coluna_temporal].duplicated().any():
        raise ValueError(
            "O DataFrame de referência contém instantes "
            "temporais duplicados."
        )

    if interpolar[coluna_temporal].duplicated().any():
        raise ValueError(
            "O DataFrame a interpolar contém instantes "
            "temporais duplicados."
        )

    primeiro_instante_interpolar = (
        interpolar[coluna_temporal].iloc[0]
    )

    ultimo_instante_interpolar = (
        interpolar[coluna_temporal].iloc[-1]
    )

    # --------------------------------------------------------
    # Seleção dos instantes de referência dentro do domínio
    # em que a interpolação é matematicamente válida
    # --------------------------------------------------------

    referencia = referencia.loc[
        referencia[coluna_temporal].between(
            primeiro_instante_interpolar,
            ultimo_instante_interpolar,
            inclusive="both",
        )
    ].copy()

    if referencia.empty:
        raise ValueError(
            "Os dois conjuntos de dados não possuem "
            "um intervalo temporal comum."
        )

    # --------------------------------------------------------
    # Construção do índice conjunto
    #
    # Importante:
    # O DataFrame a interpolar NÃO é previamente limitado ao
    # intervalo da referência. Desta forma são preservadas as
    # amostras imediatamente adjacentes necessárias para a
    # interpolação das fronteiras.
    # --------------------------------------------------------

    indice_referencia = pd.DatetimeIndex(
        referencia[coluna_temporal]
    )

    indice_interpolar = pd.DatetimeIndex(
        interpolar[coluna_temporal]
    )

    indice_combinado = (
        indice_interpolar
        .union(indice_referencia)
        .sort_values()
    )

    dados_interpolados = (
        interpolar
        .set_index(coluna_temporal)
        .reindex(indice_combinado)
    )

    dados_interpolados[
        colunas_interpolar
    ] = (
        dados_interpolados[
            colunas_interpolar
        ]
        .interpolate(
            method="time",
            limit_area="inside",
        )
    )

    dados_interpolados = (
        dados_interpolados
        .reindex(indice_referencia)
        [colunas_interpolar]
        .copy()
    )

    dados_interpolados.index.name = (
        coluna_temporal
    )

    # --------------------------------------------------------
    # Verificação da interpolação
    # --------------------------------------------------------

    numero_valores_em_falta = int(
        dados_interpolados
        .isna()
        .sum()
        .sum()
    )

    if numero_valores_em_falta > 0:
        raise ValueError(
            "A interpolação temporal produziu "
            f"{numero_valores_em_falta} valores em falta. "
            "A sincronização não é considerada válida."
        )

    # --------------------------------------------------------
    # Junção com o DataFrame de referência
    # --------------------------------------------------------

    resultado = referencia.set_index(
        coluna_temporal
    )

    resultado = resultado.join(
        dados_interpolados,
        how="left",
    )

    resultado = (
        resultado
        .reset_index()
        .copy()
    )

    return resultado


def mostrar_resultado_validacao_temporal(
    resultado: ResultadoValidacaoTemporal,
) -> None:
    """
    Apresenta um resumo da validação temporal absoluta.
    """

    if not isinstance(
        resultado,
        ResultadoValidacaoTemporal,
    ):
        raise TypeError(
            "O objeto fornecido não é um "
            "ResultadoValidacaoTemporal."
        )

    print("\n========== VALIDAÇÃO TEMPORAL ==========")

    print(f"\nSequência válida: {resultado.valido}")

    print(
        f"Número de registos: "
        f"{resultado.numero_registos}"
    )

    print(
        f"Primeiro instante: "
        f"{resultado.primeiro_instante}"
    )

    print(
        f"Último instante: "
        f"{resultado.ultimo_instante}"
    )

    print(
        "Sequência crescente: "
        f"{resultado.monotonicamente_crescente}"
    )

    print(
        "Instantes duplicados: "
        f"{resultado.numero_instantes_duplicados}"
    )

    print(
        "Intervalo esperado: "
        f"{resultado.intervalo_esperado_s:.3f} s"
    )

    print(
        "Tolerância: "
        f"{resultado.tolerancia_s:.3f} s"
    )

    print(
        "Intervalo mediano: "
        f"{resultado.intervalo_mediano_s:.3f} s"
    )

    print(
        "Intervalo mínimo: "
        f"{resultado.intervalo_minimo_s:.3f} s"
    )

    print(
        "Intervalo máximo: "
        f"{resultado.intervalo_maximo_s:.3f} s"
    )

    print(
        "Intervalos irregulares: "
        f"{resultado.numero_intervalos_irregulares}"
    )

    print("\n========================================")


def mostrar_resultado_tempo_relativo(
    resultado: ResultadoValidacaoTempoRelativo,
) -> None:
    """
    Apresenta um resumo da validação temporal relativa.
    """

    if not isinstance(
        resultado,
        ResultadoValidacaoTempoRelativo,
    ):
        raise TypeError(
            "O objeto fornecido não é um "
            "ResultadoValidacaoTempoRelativo."
        )

    duracao_h = (
        resultado.duracao_total_s
        / 3600.0
    )

    duracao_d = (
        duracao_h
        / 24.0
    )

    print(
        "\n========== VALIDAÇÃO DO TEMPO RELATIVO =========="
    )

    print(f"\nSequência válida: {resultado.valido}")

    print(
        f"Número de registos: "
        f"{resultado.numero_registos}"
    )

    print(
        f"Primeiro tempo: "
        f"{resultado.primeiro_tempo_s:.6f} s"
    )

    print(
        f"Último tempo: "
        f"{resultado.ultimo_tempo_s:.6f} s"
    )

    print(
        f"Duração total: "
        f"{resultado.duracao_total_s:.3f} s"
    )

    print(
        f"Duração total: "
        f"{duracao_h:.2f} h ({duracao_d:.2f} dias)"
    )

    print(
        "Sequência crescente: "
        f"{resultado.monotonicamente_crescente}"
    )

    print(
        "Instantes duplicados: "
        f"{resultado.numero_instantes_duplicados}"
    )

    print(
        "Intervalo esperado: "
        f"{resultado.intervalo_esperado_s:.3f} s"
    )

    print(
        "Tolerância: "
        f"{resultado.tolerancia_s:.3f} s"
    )

    print(
        "Intervalo médio: "
        f"{resultado.intervalo_medio_s:.6f} s"
    )

    print(
        "Intervalo mediano: "
        f"{resultado.intervalo_mediano_s:.6f} s"
    )

    print(
        "Intervalo mínimo: "
        f"{resultado.intervalo_minimo_s:.6f} s"
    )

    print(
        "Intervalo máximo: "
        f"{resultado.intervalo_maximo_s:.6f} s"
    )

    print(
        "Intervalos irregulares: "
        f"{resultado.numero_intervalos_irregulares}"
    )

    print(
        "\n================================================="
    )


def exigir_sequencia_temporal_valida(
    resultado: ResultadoValidacaoTemporal,
) -> None:
    """
    Interrompe a execução quando a validação temporal absoluta
    não é aprovada.
    """

    if not isinstance(
        resultado,
        ResultadoValidacaoTemporal,
    ):
        raise TypeError(
            "O objeto fornecido não é um "
            "ResultadoValidacaoTemporal."
        )

    if resultado.valido:
        return

    problemas: list[str] = []

    if not resultado.monotonicamente_crescente:
        problemas.append(
            "a sequência temporal não é crescente"
        )

    if resultado.numero_instantes_duplicados > 0:
        problemas.append(
            "instantes temporais duplicados: "
            f"{resultado.numero_instantes_duplicados}"
        )

    if resultado.numero_intervalos_irregulares > 0:
        problemas.append(
            "intervalos de amostragem irregulares: "
            f"{resultado.numero_intervalos_irregulares}"
        )

    descricao = "; ".join(problemas)

    raise ValueError(
        "A validação temporal não foi aprovada: "
        f"{descricao}"
    )


def exigir_tempo_relativo_valido(
    resultado: ResultadoValidacaoTempoRelativo,
) -> None:
    """
    Interrompe a execução quando a validação do tempo relativo
    não é aprovada.
    """

    if not isinstance(
        resultado,
        ResultadoValidacaoTempoRelativo,
    ):
        raise TypeError(
            "O objeto fornecido não é um "
            "ResultadoValidacaoTempoRelativo."
        )

    if resultado.valido:
        return

    problemas: list[str] = []

    if not resultado.monotonicamente_crescente:
        problemas.append(
            "a sequência temporal não é estritamente crescente"
        )

    if resultado.numero_instantes_duplicados > 0:
        problemas.append(
            "instantes temporais duplicados: "
            f"{resultado.numero_instantes_duplicados}"
        )

    if resultado.numero_intervalos_irregulares > 0:
        problemas.append(
            "intervalos de amostragem fora da tolerância: "
            f"{resultado.numero_intervalos_irregulares}"
        )

    descricao = "; ".join(problemas)

    raise ValueError(
        "A validação do tempo relativo não foi aprovada: "
        f"{descricao}"
    )