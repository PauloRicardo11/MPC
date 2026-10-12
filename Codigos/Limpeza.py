"""
============================================================
Módulo: Limpeza.py

Descrição:
    Funções destinadas à limpeza controlada dos dados
    experimentais utilizados no projeto.

    O módulo permite remover colunas completamente vazias,
    eliminar linhas completamente duplicadas, selecionar
    subconjuntos de colunas, renomear variáveis e converter
    tipos de dados de forma explícita.

    Todas as operações são realizadas sobre cópias dos
    DataFrames recebidos, preservando os dados originais.

    O preenchimento, interpolação ou substituição automática
    de valores em falta não é realizado neste módulo.

Autor:
    Paulo Ribeiro

Projeto:
    Plataforma de Processamento de Dados Experimentais
============================================================
"""

from collections.abc import Mapping, Sequence

import pandas as pd


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


def remover_colunas_vazias(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Remove colunas cujos valores são todos nulos.

    A operação é realizada sobre uma cópia do DataFrame original.

    Parameters
    ----------
    dataframe:
        DataFrame a processar.

    Returns
    -------
    pandas.DataFrame
        Novo DataFrame sem colunas completamente vazias.
    """

    verificar_dataframe(dataframe)

    dataframe_limpo = dataframe.copy()

    colunas_vazias = [
        coluna
        for coluna in dataframe_limpo.columns
        if dataframe_limpo[coluna].isna().all()
    ]

    if colunas_vazias:
        dataframe_limpo = dataframe_limpo.drop(
            columns=colunas_vazias
        )

    return dataframe_limpo


def remover_linhas_duplicadas(
    dataframe: pd.DataFrame,
    manter: str | bool = "first",
) -> pd.DataFrame:
    """
    Remove linhas completamente duplicadas.

    Parameters
    ----------
    dataframe:
        DataFrame a processar.

    manter:
        Critério utilizado para manter duplicados.

        Valores possíveis:

        - ``"first"``: mantém a primeira ocorrência;
        - ``"last"``: mantém a última ocorrência;
        - ``False``: remove todas as ocorrências duplicadas.

    Returns
    -------
    pandas.DataFrame
        Novo DataFrame sem as linhas duplicadas selecionadas.

    Raises
    ------
    ValueError
        Caso o parâmetro ``manter`` seja inválido.
    """

    verificar_dataframe(dataframe)

    valores_validos = ("first", "last", False)

    if manter not in valores_validos:
        raise ValueError(
            "O parâmetro 'manter' deve ser 'first', 'last' ou False."
        )

    dataframe_limpo = (
        dataframe
        .drop_duplicates(keep=manter)
        .reset_index(drop=True)
        .copy()
    )

    return dataframe_limpo


def selecionar_colunas(
    dataframe: pd.DataFrame,
    colunas: Sequence[str],
) -> pd.DataFrame:
    """
    Seleciona explicitamente um subconjunto de colunas.

    A ordem das colunas no DataFrame resultante corresponde
    à ordem fornecida no argumento ``colunas``.

    Parameters
    ----------
    dataframe:
        DataFrame a processar.

    colunas:
        Sequência com os nomes das colunas a selecionar.

    Returns
    -------
    pandas.DataFrame
        Novo DataFrame contendo apenas as colunas selecionadas.

    Raises
    ------
    TypeError
        Caso ``colunas`` possua um tipo inválido ou contenha
        elementos que não sejam strings.

    ValueError
        Caso a sequência esteja vazia, contenha nomes repetidos
        ou alguma coluna não exista no DataFrame.
    """

    verificar_dataframe(dataframe)

    if isinstance(colunas, str) or not isinstance(
        colunas,
        Sequence,
    ):
        raise TypeError(
            "As colunas devem ser fornecidas através de uma "
            "sequência de strings."
        )

    colunas_lista = list(colunas)

    if not colunas_lista:
        raise ValueError(
            "A lista de colunas não pode estar vazia."
        )

    if not all(
        isinstance(coluna, str)
        for coluna in colunas_lista
    ):
        raise TypeError(
            "Todos os nomes das colunas devem ser strings."
        )

    if len(colunas_lista) != len(set(colunas_lista)):
        raise ValueError(
            "A lista de colunas contém nomes repetidos."
        )

    colunas_em_falta = [
        coluna
        for coluna in colunas_lista
        if coluna not in dataframe.columns
    ]

    if colunas_em_falta:
        raise ValueError(
            "As seguintes colunas não existem no DataFrame: "
            f"{colunas_em_falta}"
        )

    return dataframe.loc[:, colunas_lista].copy()


def renomear_colunas(
    dataframe: pd.DataFrame,
    mapa_nomes: Mapping[str, str],
) -> pd.DataFrame:
    """
    Renomeia colunas de forma explícita.

    Parameters
    ----------
    dataframe:
        DataFrame a processar.

    mapa_nomes:
        Mapeamento entre o nome atual e o novo nome da coluna.

    Returns
    -------
    pandas.DataFrame
        Novo DataFrame com as colunas renomeadas.

    Raises
    ------
    TypeError
        Caso o mapeamento possua um tipo inválido ou contenha
        chaves/valores que não sejam strings.

    ValueError
        Caso o mapeamento esteja vazio, contenha colunas
        inexistentes ou produza nomes de colunas duplicados.
    """

    verificar_dataframe(dataframe)

    if not isinstance(mapa_nomes, Mapping):
        raise TypeError(
            "O mapa de nomes deve ser um objeto de mapeamento."
        )

    if not mapa_nomes:
        raise ValueError(
            "O mapa de nomes não pode estar vazio."
        )

    if not all(
        isinstance(nome_atual, str)
        and isinstance(nome_novo, str)
        for nome_atual, nome_novo in mapa_nomes.items()
    ):
        raise TypeError(
            "Os nomes atuais e os novos nomes devem ser strings."
        )

    nomes_vazios = [
        nome_novo
        for nome_novo in mapa_nomes.values()
        if not nome_novo.strip()
    ]

    if nomes_vazios:
        raise ValueError(
            "Os novos nomes das colunas não podem ser vazios."
        )

    colunas_em_falta = [
        coluna
        for coluna in mapa_nomes
        if coluna not in dataframe.columns
    ]

    if colunas_em_falta:
        raise ValueError(
            "As seguintes colunas não existem no DataFrame: "
            f"{colunas_em_falta}"
        )

    dataframe_limpo = dataframe.rename(
        columns=dict(mapa_nomes)
    ).copy()

    if dataframe_limpo.columns.duplicated().any():
        duplicadas = dataframe_limpo.columns[
            dataframe_limpo.columns.duplicated()
        ].astype(str).tolist()

        raise ValueError(
            "O renomeamento originou nomes de colunas duplicados: "
            f"{duplicadas}"
        )

    return dataframe_limpo


def converter_tipos(
    dataframe: pd.DataFrame,
    tipos: Mapping[str, str | type],
) -> pd.DataFrame:
    """
    Converte explicitamente os tipos de dados de determinadas
    colunas.

    A função utiliza ``pandas.Series.astype`` e não realiza
    conversões silenciosas ou preenchimento de valores inválidos.

    Parameters
    ----------
    dataframe:
        DataFrame a processar.

    tipos:
        Mapeamento entre nomes de colunas e tipos pretendidos.

    Returns
    -------
    pandas.DataFrame
        Novo DataFrame com os tipos convertidos.

    Raises
    ------
    TypeError
        Caso ``tipos`` não seja um objeto de mapeamento.

    ValueError
        Caso o mapeamento esteja vazio, contenha colunas
        inexistentes ou não seja possível efetuar alguma conversão.
    """

    verificar_dataframe(dataframe)

    if not isinstance(tipos, Mapping):
        raise TypeError(
            "Os tipos devem ser fornecidos através de um "
            "objeto de mapeamento."
        )

    if not tipos:
        raise ValueError(
            "O mapeamento de tipos não pode estar vazio."
        )

    if not all(
        isinstance(coluna, str)
        for coluna in tipos
    ):
        raise TypeError(
            "Todos os nomes das colunas devem ser strings."
        )

    colunas_em_falta = [
        coluna
        for coluna in tipos
        if coluna not in dataframe.columns
    ]

    if colunas_em_falta:
        raise ValueError(
            "As seguintes colunas não existem no DataFrame: "
            f"{colunas_em_falta}"
        )

    dataframe_limpo = dataframe.copy()

    for coluna, tipo in tipos.items():
        try:
            dataframe_limpo[coluna] = (
                dataframe_limpo[coluna].astype(tipo)
            )

        except (TypeError, ValueError) as erro:
            raise ValueError(
                f"Não foi possível converter a coluna '{coluna}' "
                f"para o tipo '{tipo}'."
            ) from erro

    return dataframe_limpo


def remover_linhas_com_valores_em_falta(
    dataframe: pd.DataFrame,
    colunas: Sequence[str] | None = None,
) -> pd.DataFrame:
    """
    Remove linhas que apresentam valores em falta.

    A remoção pode ser aplicada a todas as colunas ou apenas
    a um subconjunto explicitamente indicado.

    Esta operação deve ser utilizada apenas quando a remoção
    dos registos estiver metodologicamente justificada.

    Parameters
    ----------
    dataframe:
        DataFrame a processar.

    colunas:
        Colunas utilizadas para identificar valores em falta.
        Quando ``None``, são consideradas todas as colunas.

    Returns
    -------
    pandas.DataFrame
        Novo DataFrame sem as linhas selecionadas.

    Raises
    ------
    TypeError
        Caso ``colunas`` possua um tipo inválido.

    ValueError
        Caso alguma coluna indicada não exista no DataFrame.
    """

    verificar_dataframe(dataframe)

    if colunas is None:
        subconjunto = None

    else:
        if isinstance(colunas, str) or not isinstance(
            colunas,
            Sequence,
        ):
            raise TypeError(
                "As colunas devem ser fornecidas através de uma "
                "sequência de strings ou None."
            )

        subconjunto = list(colunas)

        if not all(
            isinstance(coluna, str)
            for coluna in subconjunto
        ):
            raise TypeError(
                "Todos os nomes das colunas devem ser strings."
            )

        colunas_em_falta = [
            coluna
            for coluna in subconjunto
            if coluna not in dataframe.columns
        ]

        if colunas_em_falta:
            raise ValueError(
                "As seguintes colunas não existem no DataFrame: "
                f"{colunas_em_falta}"
            )

    dataframe_limpo = (
        dataframe
        .dropna(subset=subconjunto)
        .reset_index(drop=True)
        .copy()
    )

    return dataframe_limpo