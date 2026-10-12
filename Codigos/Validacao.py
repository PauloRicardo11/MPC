"""
============================================================
Módulo: Validacao.py

Descrição:
    Funções destinadas à validação estrutural e inicial dos
    dados experimentais importados.

    Este módulo verifica a integridade básica dos DataFrames,
    nomeadamente a existência de dados, colunas obrigatórias,
    nomes de colunas duplicados, valores em falta e linhas
    duplicadas.

    As operações de limpeza, transformação e exportação são
    realizadas noutros módulos da plataforma.

Autor:
    Paulo Ribeiro

Projeto:
    Plataforma de Processamento de Dados Experimentais
============================================================
"""

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class ResultadoValidacao:
    """
    Resultado estruturado da validação de um DataFrame.

    Attributes
    ----------
    valido:
        Indica se o DataFrame cumpre os critérios definidos.

    numero_linhas:
        Número total de linhas.

    numero_colunas:
        Número total de colunas.

    colunas_em_falta:
        Colunas obrigatórias não encontradas.

    colunas_duplicadas:
        Nomes de colunas repetidos.

    numero_linhas_duplicadas:
        Número de linhas completamente duplicadas.

    valores_em_falta:
        Número de valores em falta por coluna.

    linhas_duplicadas_permitidas:
        Indica se as linhas duplicadas foram aceites.

    valores_em_falta_permitidos:
        Indica se os valores em falta foram aceites.
    """

    valido: bool
    numero_linhas: int
    numero_colunas: int
    colunas_em_falta: tuple[str, ...]
    colunas_duplicadas: tuple[str, ...]
    numero_linhas_duplicadas: int
    valores_em_falta: dict[str, int]
    linhas_duplicadas_permitidas: bool
    valores_em_falta_permitidos: bool


def verificar_dataframe(
    dataframe: pd.DataFrame,
) -> None:
    """
    Verifica se o objeto é um DataFrame não vazio.

    Parameters
    ----------
    dataframe:
        Objeto a verificar.

    Raises
    ------
    TypeError
        Caso o objeto não seja um DataFrame.

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


def obter_colunas_duplicadas(
    dataframe: pd.DataFrame,
) -> tuple[str, ...]:
    """
    Identifica nomes de colunas duplicados.

    Parameters
    ----------
    dataframe:
        DataFrame a analisar.

    Returns
    -------
    tuple[str, ...]
        Nomes das colunas duplicadas.
    """

    verificar_dataframe(dataframe)

    colunas_duplicadas = dataframe.columns[
        dataframe.columns.duplicated()
    ].astype(str).tolist()

    return tuple(colunas_duplicadas)


def obter_colunas_em_falta(
    dataframe: pd.DataFrame,
    colunas_obrigatorias: list[str] | tuple[str, ...] | None,
) -> tuple[str, ...]:
    """
    Identifica colunas obrigatórias ausentes.

    Parameters
    ----------
    dataframe:
        DataFrame a analisar.

    colunas_obrigatorias:
        Colunas que devem existir. Quando é None, esta verificação
        não é efetuada.

    Returns
    -------
    tuple[str, ...]
        Colunas obrigatórias não encontradas.

    Raises
    ------
    TypeError
        Caso as colunas obrigatórias possuam um tipo inválido.
    """

    verificar_dataframe(dataframe)

    if colunas_obrigatorias is None:
        return tuple()

    if not isinstance(colunas_obrigatorias, (list, tuple)):
        raise TypeError(
            "As colunas obrigatórias devem ser fornecidas "
            "como lista, tuplo ou None."
        )

    if not all(
        isinstance(coluna, str)
        for coluna in colunas_obrigatorias
    ):
        raise TypeError(
            "Todos os nomes das colunas obrigatórias "
            "devem ser strings."
        )

    colunas_em_falta = [
        coluna
        for coluna in colunas_obrigatorias
        if coluna not in dataframe.columns
    ]

    return tuple(colunas_em_falta)


def contar_linhas_duplicadas(
    dataframe: pd.DataFrame,
) -> int:
    """
    Conta linhas completamente duplicadas.

    Parameters
    ----------
    dataframe:
        DataFrame a analisar.

    Returns
    -------
    int
        Número de linhas duplicadas.
    """

    verificar_dataframe(dataframe)

    return int(dataframe.duplicated().sum())


def contar_valores_em_falta(
    dataframe: pd.DataFrame,
) -> dict[str, int]:
    """
    Conta valores em falta por coluna.

    Parameters
    ----------
    dataframe:
        DataFrame a analisar.

    Returns
    -------
    dict[str, int]
        Número de valores em falta por coluna.
    """

    verificar_dataframe(dataframe)

    valores_em_falta = dataframe.isna().sum()

    return {
        str(coluna): int(numero)
        for coluna, numero in valores_em_falta.items()
    }


def validar_dataframe(
    dataframe: pd.DataFrame,
    colunas_obrigatorias: list[str] | tuple[str, ...] | None = None,
    permitir_linhas_duplicadas: bool = True,
    permitir_valores_em_falta: bool = True,
) -> ResultadoValidacao:
    """
    Valida a estrutura básica de um DataFrame.

    Parameters
    ----------
    dataframe:
        DataFrame a validar.

    colunas_obrigatorias:
        Colunas que devem existir no DataFrame.

    permitir_linhas_duplicadas:
        Define se linhas duplicadas são admissíveis.

    permitir_valores_em_falta:
        Define se valores em falta são admissíveis.

    Returns
    -------
    ResultadoValidacao
        Resultado estruturado da validação.

    Raises
    ------
    TypeError
        Caso os parâmetros possuam tipos inválidos.

    ValueError
        Caso o DataFrame esteja vazio.
    """

    verificar_dataframe(dataframe)

    if not isinstance(permitir_linhas_duplicadas, bool):
        raise TypeError(
            "O parâmetro 'permitir_linhas_duplicadas' "
            "deve ser booleano."
        )

    if not isinstance(permitir_valores_em_falta, bool):
        raise TypeError(
            "O parâmetro 'permitir_valores_em_falta' "
            "deve ser booleano."
        )

    colunas_em_falta = obter_colunas_em_falta(
        dataframe=dataframe,
        colunas_obrigatorias=colunas_obrigatorias,
    )

    colunas_duplicadas = obter_colunas_duplicadas(dataframe)

    numero_linhas_duplicadas = contar_linhas_duplicadas(dataframe)

    valores_em_falta = contar_valores_em_falta(dataframe)

    existem_valores_em_falta = any(
        quantidade > 0
        for quantidade in valores_em_falta.values()
    )

    valido = (
        not colunas_em_falta
        and not colunas_duplicadas
        and (
            permitir_linhas_duplicadas
            or numero_linhas_duplicadas == 0
        )
        and (
            permitir_valores_em_falta
            or not existem_valores_em_falta
        )
    )

    return ResultadoValidacao(
        valido=valido,
        numero_linhas=int(dataframe.shape[0]),
        numero_colunas=int(dataframe.shape[1]),
        colunas_em_falta=colunas_em_falta,
        colunas_duplicadas=colunas_duplicadas,
        numero_linhas_duplicadas=numero_linhas_duplicadas,
        valores_em_falta=valores_em_falta,
        linhas_duplicadas_permitidas=permitir_linhas_duplicadas,
        valores_em_falta_permitidos=permitir_valores_em_falta,
    )


def exigir_dataframe_valido(
    resultado: ResultadoValidacao,
) -> None:
    """
    Interrompe a execução quando a validação não é aprovada.

    Parameters
    ----------
    resultado:
        Resultado devolvido por validar_dataframe.

    Raises
    ------
    TypeError
        Caso o objeto não seja ResultadoValidacao.

    ValueError
        Caso a validação não tenha sido aprovada.
    """

    if not isinstance(resultado, ResultadoValidacao):
        raise TypeError(
            "O objeto fornecido não é um ResultadoValidacao."
        )

    if resultado.valido:
        return

    problemas: list[str] = []

    if resultado.colunas_em_falta:
        problemas.append(
            "colunas obrigatórias em falta: "
            f"{list(resultado.colunas_em_falta)}"
        )

    if resultado.colunas_duplicadas:
        problemas.append(
            "nomes de colunas duplicados: "
            f"{list(resultado.colunas_duplicadas)}"
        )

    if (
        resultado.numero_linhas_duplicadas > 0
        and not resultado.linhas_duplicadas_permitidas
    ):
        problemas.append(
            "linhas duplicadas: "
            f"{resultado.numero_linhas_duplicadas}"
        )

    colunas_com_valores_em_falta = {
        coluna: quantidade
        for coluna, quantidade in resultado.valores_em_falta.items()
        if quantidade > 0
    }

    if (
        colunas_com_valores_em_falta
        and not resultado.valores_em_falta_permitidos
    ):
        problemas.append(
            "valores em falta: "
            f"{colunas_com_valores_em_falta}"
        )

    descricao = "; ".join(problemas)

    raise ValueError(
        f"A validação do DataFrame não foi aprovada: {descricao}"
    )


def mostrar_resultado_validacao(
    resultado: ResultadoValidacao,
) -> None:
    """
    Apresenta um resumo estruturado da validação.

    Parameters
    ----------
    resultado:
        Resultado devolvido por validar_dataframe.

    Raises
    ------
    TypeError
        Caso o objeto não seja ResultadoValidacao.
    """

    if not isinstance(resultado, ResultadoValidacao):
        raise TypeError(
            "O objeto fornecido não é um ResultadoValidacao."
        )

    total_valores_em_falta = sum(
        resultado.valores_em_falta.values()
    )

    print("\n========== RESULTADO DA VALIDAÇÃO ==========")
    print(f"\nDataFrame válido: {resultado.valido}")
    print(f"Número de linhas: {resultado.numero_linhas}")
    print(f"Número de colunas: {resultado.numero_colunas}")

    print(
        "Colunas obrigatórias em falta: "
        f"{list(resultado.colunas_em_falta)}"
    )

    print(
        "Nomes de colunas duplicados: "
        f"{list(resultado.colunas_duplicadas)}"
    )

    print(
        "Número de linhas duplicadas: "
        f"{resultado.numero_linhas_duplicadas}"
    )

    print(
        "Número total de valores em falta: "
        f"{total_valores_em_falta}"
    )

    print("\n============================================")


def mostrar_resumo(
    dataframe: pd.DataFrame,
    numero_linhas: int = 5,
) -> None:
    """
    Apresenta um resumo inicial do DataFrame.

    A função destina-se à visualização no notebook e não altera
    o DataFrame.

    Parameters
    ----------
    dataframe:
        DataFrame a analisar.

    numero_linhas:
        Número de linhas iniciais a apresentar.

    Raises
    ------
    TypeError
        Caso os parâmetros possuam tipos inválidos.

    ValueError
        Caso o número de linhas seja inferior ou igual a zero.
    """

    verificar_dataframe(dataframe)

    if not isinstance(numero_linhas, int):
        raise TypeError(
            "O número de linhas deve ser um valor inteiro."
        )

    if numero_linhas <= 0:
        raise ValueError(
            "O número de linhas deve ser superior a zero."
        )

    print("\n========== RESUMO DOS DADOS ==========")

    print(f"\nNúmero de linhas: {dataframe.shape[0]}")
    print(f"Número de colunas: {dataframe.shape[1]}")

    print("\nNomes das colunas:")
    for coluna in dataframe.columns:
        print(f"  - {coluna}")

    print("\nTipos de dados:")
    print(dataframe.dtypes)

    print("\nValores em falta por coluna:")
    print(dataframe.isna().sum())

    print(
        "\nNúmero de linhas duplicadas: "
        f"{int(dataframe.duplicated().sum())}"
    )

    print(f"\nPrimeiras {numero_linhas} linhas:")
    print(dataframe.head(numero_linhas))

    print("\n======================================")