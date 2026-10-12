"""
============================================================
Módulo: Configuracao.py

Descrição:
    Configuração centralizada dos caminhos e da estrutura
    da Plataforma de Processamento de Dados Experimentais.

    Este módulo permite localizar automaticamente a raiz do
    projeto, construir os caminhos utilizados pela plataforma,
    validar a estrutura de diretórios e criar apenas as pastas
    destinadas ao processamento e armazenamento de resultados.

    As pastas Original são consideradas áreas protegidas e nunca
    são criadas nem alteradas automaticamente pela plataforma.

Autor:
    Paulo Ribeiro

Projeto:
    Plataforma de Processamento de Dados Experimentais
============================================================
"""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CaminhosProjeto:
    """
    Estrutura imutável com os principais caminhos do projeto.

    Attributes
    ----------
    raiz:
        Diretório raiz do projeto.

    codigos:
        Diretório que contém os módulos Python.

    notebooks:
        Diretório que contém os notebooks.

    aquisicao_dados:
        Diretório principal dos dados experimentais.

    temperatura_original:
        Diretório dos ficheiros originais de temperatura.

    temperatura_utilizados:
        Diretório dos ficheiros processados de temperatura.

    piso_radiante_original:
        Diretório dos ficheiros originais do analisador elétrico.

    piso_radiante_utilizados:
        Diretório dos ficheiros processados do piso radiante.

    resultados:
        Diretório destinado aos resultados numéricos.

    figuras:
        Diretório destinado às figuras geradas.

    logs:
        Diretório destinado aos ficheiros de registo.
    """

    raiz: Path
    codigos: Path
    notebooks: Path
    aquisicao_dados: Path
    temperatura_original: Path
    temperatura_utilizados: Path
    piso_radiante_original: Path
    piso_radiante_utilizados: Path
    resultados: Path
    figuras: Path
    logs: Path


def obter_raiz_projeto() -> Path:
    """
    Determina automaticamente o diretório raiz do projeto.

    A raiz é determinada a partir da localização deste módulo,
    assumindo que Configuracao.py se encontra diretamente dentro
    da pasta Codigos.

    Returns
    -------
    pathlib.Path
        Caminho absoluto para a raiz do projeto.

    Raises
    ------
    RuntimeError
        Caso o módulo não esteja localizado na estrutura esperada.
    """

    caminho_modulo = Path(__file__).resolve()
    pasta_codigos = caminho_modulo.parent
    raiz_projeto = pasta_codigos.parent

    if pasta_codigos.name != "Codigos":
        raise RuntimeError(
            "O módulo Configuracao.py deve estar localizado "
            "diretamente dentro da pasta 'Codigos'."
        )

    return raiz_projeto


def construir_caminhos() -> CaminhosProjeto:
    """
    Constrói os caminhos utilizados pela plataforma.

    Returns
    -------
    CaminhosProjeto
        Estrutura imutável com os principais diretórios do projeto.
    """

    raiz = obter_raiz_projeto()

    aquisicao_dados = raiz / "Aquisicao_Dados"
    temperatura = aquisicao_dados / "Medicao_Temperatura"
    piso_radiante = aquisicao_dados / "Medicoes_Piso_Radiante"

    return CaminhosProjeto(
        raiz=raiz,
        codigos=raiz / "Codigos",
        notebooks=raiz / "Notebooks",
        aquisicao_dados=aquisicao_dados,
        temperatura_original=temperatura / "Original",
        temperatura_utilizados=temperatura / "Utilizados",
        piso_radiante_original=piso_radiante / "Original",
        piso_radiante_utilizados=piso_radiante / "Utilizados",
        resultados=raiz / "Resultados",
        figuras=raiz / "Figuras",
        logs=raiz / "Logs",
    )


def caminho_esta_contido(
    caminho: str | Path,
    diretorio: str | Path,
) -> bool:
    """
    Verifica se um caminho está contido num diretório.

    Parameters
    ----------
    caminho:
        Caminho que se pretende verificar.

    diretorio:
        Diretório de referência.

    Returns
    -------
    bool
        True caso o caminho corresponda ao diretório de referência
        ou esteja localizado dentro dele.

    Raises
    ------
    TypeError
        Caso algum dos parâmetros possua um tipo inválido.
    """

    if not isinstance(caminho, (str, Path)):
        raise TypeError(
            "O caminho deve ser uma string ou um objeto pathlib.Path."
        )

    if not isinstance(diretorio, (str, Path)):
        raise TypeError(
            "O diretório deve ser uma string ou um objeto pathlib.Path."
        )

    caminho_resolvido = Path(caminho).expanduser().resolve()
    diretorio_resolvido = Path(diretorio).expanduser().resolve()

    return (
        caminho_resolvido == diretorio_resolvido
        or diretorio_resolvido in caminho_resolvido.parents
    )


def validar_caminho_saida(
    caminho_saida: str | Path,
    caminhos: CaminhosProjeto | None = None,
) -> Path:
    """
    Valida um caminho destinado à escrita de dados ou resultados.

    A função impede qualquer operação de escrita dentro das pastas
    que contêm os ficheiros experimentais originais.

    Parameters
    ----------
    caminho_saida:
        Caminho do ficheiro ou diretório de saída.

    caminhos:
        Estrutura de caminhos do projeto. Caso não seja fornecida,
        será construída automaticamente.

    Returns
    -------
    pathlib.Path
        Caminho absoluto e validado.

    Raises
    ------
    TypeError
        Caso os parâmetros possuam tipos inválidos.

    PermissionError
        Caso o caminho esteja localizado dentro de uma pasta Original.
    """

    if not isinstance(caminho_saida, (str, Path)):
        raise TypeError(
            "O caminho de saída deve ser uma string "
            "ou um objeto pathlib.Path."
        )

    if caminhos is None:
        caminhos = construir_caminhos()

    if not isinstance(caminhos, CaminhosProjeto):
        raise TypeError(
            "O parâmetro 'caminhos' deve ser um objeto CaminhosProjeto."
        )

    caminho_validado = Path(
        caminho_saida
    ).expanduser().resolve()

    pastas_protegidas = (
        caminhos.temperatura_original,
        caminhos.piso_radiante_original,
    )

    for pasta_original in pastas_protegidas:
        if caminho_esta_contido(
            caminho=caminho_validado,
            diretorio=pasta_original,
        ):
            raise PermissionError(
                "Operação de escrita recusada. A pasta 'Original' "
                f"está protegida: {pasta_original}"
            )

    return caminho_validado


def validar_estrutura(
    caminhos: CaminhosProjeto | None = None,
) -> None:
    """
    Valida os diretórios obrigatórios da plataforma.

    As pastas Original têm de existir previamente e nunca são
    criadas automaticamente.

    Parameters
    ----------
    caminhos:
        Estrutura de caminhos do projeto. Caso não seja fornecida,
        será construída automaticamente.

    Raises
    ------
    TypeError
        Caso o parâmetro caminhos possua um tipo inválido.

    FileNotFoundError
        Caso uma pasta obrigatória não exista.

    NotADirectoryError
        Caso um caminho obrigatório não corresponda a um diretório.
    """

    if caminhos is None:
        caminhos = construir_caminhos()

    if not isinstance(caminhos, CaminhosProjeto):
        raise TypeError(
            "O parâmetro 'caminhos' deve ser um objeto CaminhosProjeto."
        )

    pasta_temperatura = caminhos.temperatura_original.parent
    pasta_piso_radiante = caminhos.piso_radiante_original.parent

    pastas_obrigatorias = {
        "Codigos": caminhos.codigos,
        "Aquisicao_Dados": caminhos.aquisicao_dados,
        "Medicao_Temperatura": pasta_temperatura,
        "Medicao_Temperatura/Original":
            caminhos.temperatura_original,
        "Medicoes_Piso_Radiante": pasta_piso_radiante,
        "Medicoes_Piso_Radiante/Original":
            caminhos.piso_radiante_original,
    }

    for designacao, caminho in pastas_obrigatorias.items():
        if not caminho.exists():
            raise FileNotFoundError(
                f"A pasta obrigatória '{designacao}' não existe: "
                f"{caminho}"
            )

        if not caminho.is_dir():
            raise NotADirectoryError(
                f"O caminho associado a '{designacao}' não é "
                f"um diretório: {caminho}"
            )


def criar_pastas_trabalho(
    caminhos: CaminhosProjeto | None = None,
) -> None:
    """
    Cria as pastas de trabalho utilizadas pela plataforma.

    As pastas Original nunca são incluídas nesta operação.

    Parameters
    ----------
    caminhos:
        Estrutura de caminhos do projeto. Caso não seja fornecida,
        será construída automaticamente.

    Raises
    ------
    TypeError
        Caso o parâmetro caminhos possua um tipo inválido.
    """

    if caminhos is None:
        caminhos = construir_caminhos()

    if not isinstance(caminhos, CaminhosProjeto):
        raise TypeError(
            "O parâmetro 'caminhos' deve ser um objeto CaminhosProjeto."
        )

    pastas_trabalho = (
        caminhos.notebooks,
        caminhos.temperatura_utilizados,
        caminhos.piso_radiante_utilizados,
        caminhos.resultados,
        caminhos.figuras,
        caminhos.logs,
    )

    for pasta in pastas_trabalho:
        caminho_validado = validar_caminho_saida(
            caminho_saida=pasta,
            caminhos=caminhos,
        )

        caminho_validado.mkdir(
            parents=True,
            exist_ok=True,
        )


def iniciar_plataforma() -> CaminhosProjeto:
    """
    Inicializa a estrutura da plataforma.

    A sequência de inicialização é:

    1. construção dos caminhos;
    2. validação das pastas obrigatórias;
    3. criação das pastas de trabalho.

    Returns
    -------
    CaminhosProjeto
        Estrutura imutável com os caminhos do projeto.
    """

    caminhos = construir_caminhos()

    validar_estrutura(caminhos)
    criar_pastas_trabalho(caminhos)

    return caminhos