def retorno_simple(precio_inicial, precio_final):
    """
    Calcula el retorno simple de una inversión basado en el precio inicial y el precio final.

    El retorno simple es una métrica financiera que indica el cambio proporcional en el valor
    de una inversión, expresado como una fracción. Se calcula dividiendo el precio final entre 
    el precio inicial y restando 1.

    Parameters
    ----------
    precio_inicial : int o float
        El precio inicial de la inversión. Debe ser un valor numérico mayor a cero.
    precio_final : int o float
        El precio final de la inversión. Debe ser un valor numérico mayor o igual a cero.

    Returns
    -------
    float
        El retorno simple de la inversión, calculado como una fracción decimal. Por ejemplo,
        un valor de 0.2 indica un retorno del 20%.

    Raises
    ------
    TypeError
        Si `precio_inicial` o `precio_final` no son valores numéricos (int o float).

    Examples
    --------
    >>> retorno_simple(100, 120)
    0.2

    >>> retorno_simple(100, 80)
    -0.2

    """

    # if not isinstance(precio_inicial, (int, float)) or not isinstance(precio_final, (int, float)):
    #     raise TypeError("Los parámetros 'precio_inicial' y/o 'precio_final' deben de ser valores numéricos")
    verificar_valor_numerico(precio_inicial, precio_final)
    retorno = (precio_final / precio_inicial) - 1

    return retorno


def retorno_compuesto(valores):
    """
    Calcula el retorno compuesto de una inversión basado en una lista de valores de precios sucesivos.

    El retorno compuesto es una métrica financiera que mide el crecimiento acumulado de una inversión
    a lo largo de múltiples periodos. Este retorno se calcula multiplicando los retornos de cada
    periodo sucesivo y restando 1 al resultado acumulado.

    Parameters
    ----------
    valores : list of int or float
        Lista de precios sucesivos de la inversión, donde cada elemento representa el valor de la
        inversión en un periodo determinado. La lista debe contener al menos dos valores.

    Returns
    -------
    float
        El retorno compuesto de la inversión, expresado como una fracción decimal. Por ejemplo,
        un valor de 0.3 indica un retorno compuesto del 30%.

    Raises
    ------
    ValueError
        Si la lista `valores` contiene menos de dos elementos.
    TypeError
        Si algún valor en `valores` no es numérico (int o float).

    Examples
    --------
    >>> retorno_compuesto([100, 110, 121])
    0.21

    >>> retorno_compuesto([100, 90, 81])
    -0.19

    """

    if len(valores) < 2:
        raise ValueError('A la lista le faltan más valores')

    # Calcular el retorno de cada periodo y acumular el retorno compuesto
    retorno_compuesto = 1

    for i in range(1, len(valores)):
        precio_inicial = valores[i - 1]
        precio_final = valores[i]

        verificar_valor_numerico(precio_inicial, precio_final)
    
        retorno_periodo = (precio_final / precio_inicial) - 1
        retorno_compuesto *= (1 + retorno_periodo)

    retorno_compuesto = retorno_compuesto - 1

    return retorno_compuesto
    
        

def verificar_valor_numerico(valor_inicial, valor_final):
    """
    Verifica que los valores proporcionados sean numéricos.

    Esta función valida que ambos parámetros de entrada, `valor_inicial` y `valor_final`, sean
    de tipo numérico (int o float). Si alguno de los valores no es numérico, lanza una excepción
    `TypeError`.

    Parameters
    ----------
    valor_inicial : int or float
        El valor inicial a verificar. Debe ser numérico.
    valor_final : int or float
        El valor final a verificar. Debe ser numérico.

    Raises
    ------
    TypeError
        Si `valor_inicial` o `valor_final` no son valores numéricos (int o float).

    Examples
    --------
    >>> verificar_valor_numerico(100, 120)
    # No se lanza ninguna excepción

    >>> verificar_valor_numerico(100, "120")
    TypeError: Los parámetros 'precio_inicial' y/o 'precio_final' deben de ser valores numéricos

    """

    if not isinstance(valor_inicial, (int, float)) or not isinstance(valor_final, (int, float)):
            raise TypeError("Los parámetros 'precio_inicial' y/o 'precio_final' deben de ser valores numéricos")
    











































